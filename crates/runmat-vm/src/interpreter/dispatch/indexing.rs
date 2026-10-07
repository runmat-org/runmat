use runmat_runtime::builtins::common::tensor::{tensor_value_f64, tensor_values_f64_cow};
use runmat_runtime::call::arguments::expand_brace_values;
use runmat_runtime::indexing::plan::{
    build_assignment_plan, build_index_plan, build_sparse_assignment_plan, IndexPlan,
};
use runmat_runtime::indexing::read_linear as idx_read_linear;
use runmat_runtime::indexing::read_slice as idx_read_slice;
use runmat_runtime::indexing::selectors::{
    build_assignment_selectors, build_scalar_selectors, build_slice_selectors,
    index_scalar_from_value, SliceSelector,
};
use runmat_runtime::indexing::write_linear as idx_write_linear;
use runmat_runtime::indexing::write_slice as idx_write_slice;
use runmat_runtime::object::dispatch::{
    invoke_resolved_object_index_path_method_in_context, resolve_object_index_protocol,
};
use runmat_runtime::object::indexing::{
    ObjectIndexComponent, ObjectIndexOp, ObjectIndexSelector, ObjectSubscript, ObjectSubscriptPath,
};
use runmat_runtime::sequence::ResolveValueSequence;
use runmat_runtime::{build_runtime_error, RuntimeError};
use runmat_value::{CellArray, IntValue, IntegerStorage, SymbolicExpr, Tensor, Value};

mod scalar_read_and_cell;
mod scalar_store;
mod slice_read;
mod slice_store;

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

fn resolve_single_sequence(sequence: runmat_value::ValueSequence) -> Result<Value, RuntimeError> {
    let mut values = sequence.resolve(
        runmat_types::SequenceUse::RequireSingle,
        runmat_runtime::sequence::SequenceResolutionContext::default(),
    )?;
    Ok(values.remove(0))
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
/// exact storage. This keeps slice stores on the same path as
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
        let logical = runmat_value::LogicalArray::new(logical_data, t.shape().to_vec())
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

fn object_slice_selector(
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    numeric: &[Value],
) -> Result<ObjectIndexSelector, RuntimeError> {
    if end_mask != 0 {
        return Err(crate::interpreter::errors::mex(
            "ContextualEndRequiresSubscriptPath",
            "object end must be evaluated through the contextual subscript path",
        ));
    }
    let mut values = numeric.iter();
    let mut components = Vec::with_capacity(dims);
    for dimension in 0..dims {
        if colon_mask & (1_u32 << dimension) != 0 {
            components.push(ObjectIndexComponent::Colon);
        } else {
            let value = values.next().ok_or_else(|| {
                crate::interpreter::errors::mex(
                    "MalformedObjectSelector",
                    "object selector is missing an indexed component",
                )
            })?;
            components.push(ObjectIndexComponent::Value(value.clone()));
        }
    }
    if values.next().is_some() {
        return Err(crate::interpreter::errors::mex(
            "MalformedObjectSelector",
            "object selector contains excess indexed components",
        ));
    }
    Ok(ObjectIndexSelector::IndexValues { components })
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
    let protocol = match op {
        ObjectIndexOp::Subsref => runmat_runtime::object::protocol::ObjectProtocol::Subsref,
        ObjectIndexOp::Subsasgn => runmat_runtime::object::protocol::ObjectProtocol::Subsasgn,
    };
    match runmat_runtime::object::protocol::resolve_object_protocol(
        base,
        protocol,
        &runmat_runtime::object::protocol::ObjectAccessContext::default(),
    ) {
        Ok(runmat_runtime::object::protocol::ProtocolResolution::Method(_)) => return None,
        Err(error) => return Some(error),
        Ok(runmat_runtime::object::protocol::ProtocolResolution::DefaultIndexing) => {}
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

async fn assign_object_scalar_indices(
    runtime: &runmat_runtime::context::RuntimeContext,
    base: Value,
    indices: Vec<usize>,
    rhs: Value,
    delete: bool,
) -> Result<Value, RuntimeError> {
    let _roots = crate::call::root_values([&base, &rhs])?;
    let resolution = resolve_object_index_protocol(&base, ObjectIndexOp::Subsasgn, None)?;
    if let runmat_runtime::object::protocol::ProtocolResolution::Method(method) = resolution {
        let path = ObjectSubscriptPath::single(ObjectSubscript::parentheses(
            ObjectIndexSelector::ScalarIndices { indices },
        ));
        return runmat_runtime::object::protocol::invoke_resolved_object_assignment_in_context(
            runtime,
            &method,
            base,
            path,
            runmat_runtime::call::arguments::adapt_legacy_builtin_result(rhs)?.into_values(),
        )
        .await;
    }
    runmat_runtime::indexing::object::assign_scalar_indices(base, &indices, rhs, delete)
}

async fn invoke_object_brace_read(
    runtime: &runmat_runtime::context::RuntimeContext,
    base: Value,
    indices: Vec<Value>,
    requested_outputs: usize,
) -> Result<Value, RuntimeError> {
    let _roots = crate::call::root_values(indices.iter().chain(std::iter::once(&base)))?;
    let resolution = resolve_object_index_protocol(&base, ObjectIndexOp::Subsref, None)?;
    let path =
        ObjectSubscriptPath::single(ObjectSubscript::braces(ObjectIndexSelector::IndexValues {
            components: indices.into_iter().map(Into::into).collect(),
        }));
    let sequence = invoke_resolved_object_index_path_method_in_context(
        runtime,
        &resolution,
        base,
        path,
        requested_outputs,
    )
    .await?;
    resolve_single_sequence(sequence)
}

async fn invoke_object_brace_assignment(
    runtime: &runmat_runtime::context::RuntimeContext,
    base: Value,
    indices: Vec<Value>,
    rhs: Value,
) -> Result<Value, RuntimeError> {
    let _roots = crate::call::root_values(indices.iter().chain([&base, &rhs]))?;
    let resolution = resolve_object_index_protocol(&base, ObjectIndexOp::Subsasgn, None)?;
    let runmat_runtime::object::protocol::ProtocolResolution::Method(method) = resolution else {
        return Err(crate::interpreter::errors::mex(
            "MissingSubsasgn",
            "class does not define subsasgn for brace assignment",
        ));
    };
    let path =
        ObjectSubscriptPath::single(ObjectSubscript::braces(ObjectIndexSelector::IndexValues {
            components: indices.into_iter().map(Into::into).collect(),
        }));
    runmat_runtime::object::protocol::invoke_resolved_object_assignment_in_context(
        runtime,
        &method,
        base,
        path,
        runmat_runtime::call::arguments::adapt_legacy_builtin_result(rhs)?.into_values(),
    )
    .await
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

async fn resolve_cell_indices(values: &[Value]) -> Result<Vec<usize>, RuntimeError> {
    let selectors = build_scalar_selectors(values)
        .await
        .map_err(map_cell_scalar_selector_error)?;
    let mut indices = Vec::with_capacity(selectors.len());
    for selector in selectors {
        match selector {
            SliceSelector::Scalar(index) => indices.push(index),
            _ => {
                return Err(crate::interpreter::errors::mex(
                    "CellIndexType",
                    "Unsupported cell index type",
                ))
            }
        }
    }
    Ok(indices)
}

fn gather_cell_with_plan(
    ca: &CellArray,
    plan: &runmat_runtime::indexing::plan::IndexPlan,
) -> Result<Value, RuntimeError> {
    let indices: Vec<usize> = plan.indices.iter().map(|idx| (*idx as usize) + 1).collect();
    runmat_runtime::object::cell::gather_cell_paren_linear_indices(ca, &indices, &plan.output_shape)
}

fn gather_struct_array_with_plan(
    array: &runmat_value::StructArray,
    plan: &runmat_runtime::indexing::plan::IndexPlan,
) -> Result<Value, RuntimeError> {
    runmat_runtime::indexing::structure::read_with_plan(array, plan)
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

enum BraceIndexOperation {
    ReadSingle,
    Expand { out_count: usize },
    List,
    Store { rhs: Value },
}

enum BraceIndexOutcome {
    Value(Value),
    Expanded(Vec<Value>),
}

enum BraceOutcomeExpectation {
    SingleValue { invalid_message: &'static str },
    ExpandedValues { invalid_message: &'static str },
}

struct BraceStackRequest {
    num_indices: usize,
    operation: BraceIndexOperation,
    expectation: BraceOutcomeExpectation,
}

async fn execute_brace_operation(
    base: Value,
    raw_indices: &[Value],
    operation: BraceIndexOperation,
    runtime: &runmat_runtime::context::RuntimeContext,
) -> Result<BraceIndexOutcome, RuntimeError> {
    let _roots = crate::call::root_values(raw_indices.iter().chain(std::iter::once(&base)))?;
    let _rhs_roots = match &operation {
        BraceIndexOperation::Store { rhs } => Some(crate::call::root_values([rhs])?),
        _ => None,
    };
    if matches!(
        &base,
        Value::FunctionHandle(_)
            | Value::ExternalFunctionHandle(_)
            | Value::MethodFunctionHandle(_)
            | Value::BoundFunctionHandle { .. }
            | Value::Closure(_)
    ) {
        return Err(crate::interpreter::errors::mex(
            "UnsupportedFunctionHandleSelector",
            "Function handle call does not support brace selector syntax",
        ));
    }
    match operation {
        BraceIndexOperation::ReadSingle => {
            let value = match base {
                Value::Object(obj) => {
                    if let Some(err) = missing_member_index_overload_error(
                        &Value::Object(obj.clone()),
                        ObjectIndexOp::Subsref,
                    ) {
                        return Err(err);
                    }
                    invoke_object_brace_read(runtime, Value::Object(obj), raw_indices.to_vec(), 1)
                        .await?
                }
                Value::HandleObject(handle) => {
                    if let Some(err) = missing_member_index_overload_error(
                        &Value::HandleObject(handle.clone()),
                        ObjectIndexOp::Subsref,
                    ) {
                        return Err(err);
                    }
                    invoke_object_brace_read(
                        runtime,
                        Value::HandleObject(handle),
                        raw_indices.to_vec(),
                        1,
                    )
                    .await?
                }
                Value::Cell(ca) => {
                    let indices = resolve_cell_indices(raw_indices).await?;
                    runmat_runtime::object::cell::index_cell_value(&ca, &indices)?
                }
                Value::Composite(handle) => {
                    runmat_runtime::parallel::composite::read_composite_entry(
                        runtime,
                        *handle,
                        raw_indices,
                    )
                    .await?
                }
                _ => {
                    return Err(crate::interpreter::errors::mex(
                        "CellIndexingOnNonCell",
                        "Cell indexing on non-cell",
                    ))
                }
            };
            Ok(BraceIndexOutcome::Value(value))
        }
        BraceIndexOperation::Expand { out_count } => {
            if let Value::Composite(handle) = base {
                if out_count != 1 {
                    return Err(crate::interpreter::errors::mex(
                        "CompositeOutputArity",
                        "one Composite brace selection produces exactly one value",
                    ));
                }
                return Ok(BraceIndexOutcome::Expanded(vec![
                    runmat_runtime::parallel::composite::read_composite_entry(
                        runtime,
                        *handle,
                        raw_indices,
                    )
                    .await?,
                ]));
            }
            let values = expand_brace_values(base, raw_indices, Some(out_count)).await?;
            Ok(BraceIndexOutcome::Expanded(values))
        }
        BraceIndexOperation::List => {
            if let Value::Composite(handle) = base {
                return runmat_runtime::parallel::composite::read_composite_entry(
                    runtime,
                    *handle,
                    raw_indices,
                )
                .await
                .map(|value| BraceIndexOutcome::Expanded(vec![value]));
            }
            if !matches!(
                base,
                Value::Cell(_) | Value::Object(_) | Value::HandleObject(_)
            ) {
                return Err(crate::interpreter::errors::mex(
                    "CellIndexingOnNonCell",
                    "Cell indexing on non-cell",
                ));
            }
            let values = expand_brace_values(base, raw_indices, None).await?;
            Ok(BraceIndexOutcome::Expanded(values))
        }
        BraceIndexOperation::Store { rhs } => {
            let value = match base {
                Value::Object(obj) => {
                    if let Some(err) = missing_member_index_overload_error(
                        &Value::Object(obj.clone()),
                        ObjectIndexOp::Subsasgn,
                    ) {
                        return Err(err);
                    }
                    invoke_object_brace_assignment(
                        runtime,
                        Value::Object(obj),
                        raw_indices.to_vec(),
                        rhs,
                    )
                    .await?
                }
                Value::HandleObject(handle) => {
                    if let Some(err) = missing_member_index_overload_error(
                        &Value::HandleObject(handle.clone()),
                        ObjectIndexOp::Subsasgn,
                    ) {
                        return Err(err);
                    }
                    invoke_object_brace_assignment(
                        runtime,
                        Value::HandleObject(handle),
                        raw_indices.to_vec(),
                        rhs,
                    )
                    .await?
                }
                Value::Cell(ca) => {
                    let rhs = runmat_runtime::call::arguments::adapt_legacy_builtin_result(rhs)?;
                    let rhs_kind = rhs.kind();
                    let mut rhs_values = rhs.into_values();
                    if raw_indices.len() > 1 {
                        let indices = resolve_cell_indices(raw_indices).await?;
                        match rhs_kind {
                            runmat_value::ValueSequenceKind::CommaSeparated
                                if rhs_values.len() == 1 => {
                                runmat_runtime::object::cell::assign_cell_value(
                                    ca,
                                    &indices,
                                    rhs_values.pop().unwrap_or(Value::Num(0.0)),
                                    |oldv, newv| {
                                        runmat_gc::gc_record_write(oldv, newv);
                                    },
                                )?
                            }
                            runmat_value::ValueSequenceKind::CommaSeparated => {
                                return Err(crate::interpreter::errors::mex(
                                    "CellAssignmentArityMismatch",
                                    "Cell brace assignment target count does not match source value count",
                                ))
                            }
                            runmat_value::ValueSequenceKind::Single => runmat_runtime::object::cell::assign_cell_value(
                                ca,
                                &indices,
                                rhs_values.pop().unwrap_or(Value::Num(0.0)),
                                |oldv, newv| {
                                    runmat_gc::gc_record_write(oldv, newv);
                                },
                            )?,
                        }
                    } else {
                        let positions =
                            runmat_runtime::object::cell::resolve_cell_assignment_positions(
                                &ca,
                                raw_indices,
                            )?;
                        match rhs_kind {
                            runmat_value::ValueSequenceKind::CommaSeparated => runmat_runtime::object::cell::assign_cell_value_multi(
                                ca,
                                &positions,
                                &rhs_values,
                                |oldv, newv| {
                                    runmat_gc::gc_record_write(oldv, newv);
                                },
                            )?,
                            runmat_value::ValueSequenceKind::Single if positions.len() == 1 => runmat_runtime::object::cell::assign_cell_value(
                                ca,
                                &positions,
                                rhs_values.pop().unwrap_or(Value::Num(0.0)),
                                |oldv, newv| {
                                    runmat_gc::gc_record_write(oldv, newv);
                                },
                            )?,
                            _ => {
                                return Err(crate::interpreter::errors::mex(
                                    "CellAssignmentArityMismatch",
                                    "Cell brace assignment target count does not match source value count",
                                ))
                            }
                        }
                    }
                }
                Value::Composite(_) => {
                    return Err(crate::interpreter::errors::mex(
                        "CompositeAssignmentUnsupported",
                        "Composite entries are owned by their SPMD execution and cannot be assigned from the driver",
                    ))
                }
                _ => {
                    return Err(crate::interpreter::errors::mex(
                        "CellAssignmentOnNonCell",
                        "Cell assignment on non-cell",
                    ))
                }
            };
            Ok(BraceIndexOutcome::Value(value))
        }
    }
}

async fn execute_brace_operation_from_stack(
    stack: &mut Vec<Value>,
    _vars: &mut [Value],
    request: BraceStackRequest,
    runtime: &runmat_runtime::context::RuntimeContext,
) -> Result<(), RuntimeError> {
    let raw_indices = pop_index_values(stack, request.num_indices)?;
    let base = pop_index_base(stack)?;
    let outcome = execute_brace_operation(base, &raw_indices, request.operation, runtime).await?;
    match (outcome, request.expectation) {
        (BraceIndexOutcome::Value(value), BraceOutcomeExpectation::SingleValue { .. }) => {
            stack.push(value)
        }
        (BraceIndexOutcome::Expanded(values), BraceOutcomeExpectation::ExpandedValues { .. }) => {
            for value in values {
                stack.push(value);
            }
        }
        (
            BraceIndexOutcome::Value(_),
            BraceOutcomeExpectation::ExpandedValues { invalid_message },
        )
        | (
            BraceIndexOutcome::Expanded(_),
            BraceOutcomeExpectation::SingleValue { invalid_message },
        ) => {
            return Err(crate::interpreter::errors::mex(
                "InvalidBraceIndexOutcome",
                invalid_message,
            ))
        }
    }
    Ok(())
}

pub async fn paren_index_sequence(
    runtime: &runmat_runtime::context::RuntimeContext,
    base: Value,
    raw_indices: Vec<Value>,
    requested_outputs: usize,
    function_registry: &crate::bytecode::FunctionRegistry,
) -> Result<runmat_value::ValueSequence, RuntimeError> {
    let _roots = crate::call::root_values(raw_indices.iter().chain(std::iter::once(&base)))?;
    if matches!(
        &base,
        Value::ObjectArray(_) | Value::Object(_) | Value::HandleObject(_)
    ) {
        let resolution = resolve_object_index_protocol(&base, ObjectIndexOp::Subsref, None)?;
        if matches!(
            resolution,
            runmat_runtime::object::protocol::ProtocolResolution::Method(_)
        ) {
            let path = ObjectSubscriptPath::single(ObjectSubscript::parentheses(
                ObjectIndexSelector::IndexValues {
                    components: raw_indices.into_iter().map(Into::into).collect(),
                },
            ));
            return invoke_resolved_object_index_path_method_in_context(
                runtime,
                &resolution,
                base,
                path,
                requested_outputs,
            )
            .await;
        }
    }
    if matches!(
        &base,
        Value::FunctionHandle(_)
            | Value::ExternalFunctionHandle(_)
            | Value::MethodFunctionHandle(_)
            | Value::BoundFunctionHandle { .. }
            | Value::Closure(_)
    ) {
        return match crate::call::feval::execute_feval(
            runtime,
            base,
            raw_indices,
            requested_outputs,
            function_registry,
        )
        .await?
        {
            crate::call::feval::FevalDispatch::Completed(sequence) => Ok(sequence),
        };
    }
    let value = match &base {
        Value::Foreign(reference) => {
            runmat_runtime::foreign::index_foreign_resource(reference.clone(), raw_indices).await
        }
        Value::ObjectArray(_) | Value::Object(_) | Value::HandleObject(_) => {
            let shape = runmat_runtime::indexing::object::shape(&base)
                .expect("object-like values have an object-array shape");
            let selectors =
                build_slice_selectors(raw_indices.len(), 0, 0, &raw_indices, &shape).await?;
            let plan = build_index_plan(&selectors, raw_indices.len(), &shape)?;
            runmat_runtime::indexing::object::read_with_plan(&base, &plan)
        }
        Value::Cell(ca) => {
            let selectors = build_scalar_selectors(&raw_indices).await?;
            let plan = build_index_plan(&selectors, raw_indices.len(), &ca.shape)?;
            Ok(gather_cell_with_plan(ca, &plan)?)
        }
        Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::Closure(_) => unreachable!("callable values returned above"),
        Value::Symbolic(expr) => {
            if let Some((name, parameters)) = expr.function_reference_signature() {
                apply_symbolic_function_reference(name, parameters, &raw_indices)
            } else {
                let indices = linear_index_values(&raw_indices).await?;
                if indices == [1] {
                    Ok(Value::Symbolic(expr.clone()))
                } else {
                    idx_read_linear::generic_index(&Value::Symbolic(expr.clone()), &indices).await
                }
            }
        }
        _ => {
            if let Some(shape) = runmat_runtime::indexing::value::shape(&base) {
                let selectors =
                    build_slice_selectors(raw_indices.len(), 0, 0, &raw_indices, &shape).await?;
                let plan = build_index_plan(&selectors, raw_indices.len(), &shape)?;
                runmat_runtime::indexing::value::read_with_plan(base, &plan)
            } else {
                let indices = linear_index_values(&raw_indices).await?;
                idx_read_linear::generic_index(&base, &indices).await
            }
        }
    }?;
    runmat_value::ValueSequence::single(value)
        .map_err(runmat_runtime::sequence::sequence_error_to_runtime)
}

pub async fn paren_index_value(
    runtime: &runmat_runtime::context::RuntimeContext,
    base: Value,
    raw_indices: Vec<Value>,
    requested_outputs: usize,
    function_registry: &crate::bytecode::FunctionRegistry,
) -> Result<Value, RuntimeError> {
    let sequence = paren_index_sequence(
        runtime,
        base,
        raw_indices,
        requested_outputs,
        function_registry,
    )
    .await?;
    resolve_single_sequence(sequence)
}

fn symbolic_scalar_from_value(value: &Value) -> Result<SymbolicExpr, RuntimeError> {
    match value {
        Value::Symbolic(expr) => Ok(expr.clone()),
        Value::Num(value) => Ok(SymbolicExpr::constant(*value)),
        Value::Int(value) => Ok(SymbolicExpr::constant(value.to_f64())),
        Value::Bool(value) => Ok(SymbolicExpr::constant(if *value { 1.0 } else { 0.0 })),
        Value::Tensor(tensor)
            if runmat_runtime::builtins::common::tensor::is_scalar_tensor(tensor) =>
        {
            Ok(SymbolicExpr::constant(tensor_value_f64(tensor, 0)))
        }
        Value::LogicalArray(logical) if logical.data.len() == 1 => {
            Ok(SymbolicExpr::constant(if logical.data[0] != 0 {
                1.0
            } else {
                0.0
            }))
        }
        other => Err(crate::interpreter::errors::mex(
            "InvalidSymbolicFunctionArgument",
            &format!("symbolic function arguments must be scalar values, got {other:?}"),
        )),
    }
}

fn apply_symbolic_function_reference(
    name: &str,
    parameters: &[String],
    raw_indices: &[Value],
) -> Result<Value, RuntimeError> {
    if raw_indices.len() != parameters.len() {
        return Err(crate::interpreter::errors::mex(
            "SymbolicFunctionArity",
            &format!(
                "symbolic function {name} expects {} argument(s), got {}",
                parameters.len(),
                raw_indices.len()
            ),
        ));
    }
    let args = raw_indices
        .iter()
        .map(symbolic_scalar_from_value)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Value::Symbolic(SymbolicExpr::function_call(
        name.to_string(),
        args,
    )))
}

pub async fn dispatch_indexing(
    instr: &crate::bytecode::Instr,
    stack: &mut Vec<Value>,
    vars: &mut [Value],
    function_registry: &crate::bytecode::FunctionRegistry,
    pc: usize,
    runtime: &runmat_runtime::context::RuntimeContext,
    sequence_register: &mut super::SequenceState,
) -> Result<bool, RuntimeError> {
    match instr {
        crate::bytecode::Instr::Index(_)
        | crate::bytecode::Instr::IndexCell { .. }
        | crate::bytecode::Instr::IndexCellExpand { .. }
        | crate::bytecode::Instr::IndexCellList { .. }
        | crate::bytecode::Instr::StoreIndexCell { .. }
        | crate::bytecode::Instr::StoreIndexCellDelete { .. } => {
            scalar_read_and_cell::dispatch(
                instr,
                stack,
                vars,
                function_registry,
                pc,
                runtime,
                sequence_register,
            )
            .await
        }
        crate::bytecode::Instr::StoreIndex(_) | crate::bytecode::Instr::StoreIndexDelete(_) => {
            scalar_store::dispatch(
                instr,
                stack,
                vars,
                function_registry,
                pc,
                runtime,
                sequence_register,
            )
            .await
        }
        crate::bytecode::Instr::IndexSlice(..) => {
            slice_read::dispatch(
                instr,
                stack,
                vars,
                function_registry,
                pc,
                runtime,
                sequence_register,
            )
            .await
        }
        crate::bytecode::Instr::StoreSlice(..) | crate::bytecode::Instr::StoreSliceDelete(..) => {
            slice_store::dispatch(
                instr,
                stack,
                vars,
                function_registry,
                pc,
                runtime,
                sequence_register,
            )
            .await
        }
        _ => Ok(false),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        assign_integer_scalar_with_plan, integer_scalar_tensor, map_slice_plan_error,
        symbolic_scalar_from_value,
    };
    use futures::executor::block_on;
    use runmat_runtime::indexing::plan::IndexPlan;
    use runmat_value::{
        IntValue, IntegerStorage, ObjectArray, ObjectInstance, SymbolicExpr, Tensor, Value,
    };

    #[test]
    fn integer_scalar_index_assignment_materialization_preserves_all_integer_classes() {
        macro_rules! assert_scalar_storage {
            ($value:expr, $storage:expr) => {{
                let tensor = integer_scalar_tensor($value).expect("integer scalar tensor");
                assert_eq!(tensor.integer_storage(), Some(&$storage));
            }};
        }

        assert_scalar_storage!(IntValue::I8(i8::MIN), IntegerStorage::I8(vec![i8::MIN]));
        assert_scalar_storage!(IntValue::I16(i16::MIN), IntegerStorage::I16(vec![i16::MIN]));
        assert_scalar_storage!(IntValue::I32(i32::MIN), IntegerStorage::I32(vec![i32::MIN]));
        assert_scalar_storage!(IntValue::I64(i64::MIN), IntegerStorage::I64(vec![i64::MIN]));
        assert_scalar_storage!(IntValue::U8(u8::MAX), IntegerStorage::U8(vec![u8::MAX]));
        assert_scalar_storage!(IntValue::U16(u16::MAX), IntegerStorage::U16(vec![u16::MAX]));
        assert_scalar_storage!(IntValue::U32(u32::MAX), IntegerStorage::U32(vec![u32::MAX]));
        assert_scalar_storage!(IntValue::U64(u64::MAX), IntegerStorage::U64(vec![u64::MAX]));
    }

    #[test]
    fn integer_scalar_slice_assignment_preserves_all_integer_classes() {
        macro_rules! assert_slice_assignment {
            ($value:expr, $storage:expr) => {{
                let plan = IndexPlan::new(vec![0], vec![1, 1], vec![1], 1, vec![1, 1]);
                let result = block_on(assign_integer_scalar_with_plan(
                    $value.clone(),
                    &plan,
                    &Value::Int($value),
                    false,
                ))
                .expect("integer scalar slice assignment");
                let Value::Tensor(tensor) = result else {
                    panic!("integer scalar slice assignment must return a tensor");
                };
                assert_eq!(tensor.integer_storage(), Some(&$storage));
            }};
        }

        assert_slice_assignment!(IntValue::I8(i8::MIN), IntegerStorage::I8(vec![i8::MIN]));
        assert_slice_assignment!(IntValue::I16(i16::MIN), IntegerStorage::I16(vec![i16::MIN]));
        assert_slice_assignment!(IntValue::I32(i32::MIN), IntegerStorage::I32(vec![i32::MIN]));
        assert_slice_assignment!(IntValue::I64(i64::MIN), IntegerStorage::I64(vec![i64::MIN]));
        assert_slice_assignment!(IntValue::U8(u8::MAX), IntegerStorage::U8(vec![u8::MAX]));
        assert_slice_assignment!(IntValue::U16(u16::MAX), IntegerStorage::U16(vec![u16::MAX]));
        assert_slice_assignment!(IntValue::U32(u32::MAX), IntegerStorage::U32(vec![u32::MAX]));
        assert_slice_assignment!(IntValue::U64(u64::MAX), IntegerStorage::U64(vec![u64::MAX]));
    }

    #[test]
    fn map_slice_plan_error_preserves_identifier_and_adds_context() {
        let err = crate::interpreter::errors::mex("ShapeMismatch", "shape mismatch");
        let mapped = map_slice_plan_error("slice assign", err);
        assert_eq!(mapped.identifier(), Some("RunMat:ShapeMismatch"));
        assert_eq!(mapped.message(), "slice assign: shape mismatch");
    }

    #[test]
    fn map_slice_plan_error_keeps_identifier_absent_when_missing() {
        let err = runmat_runtime::build_runtime_error("plain error").build();
        let mapped = map_slice_plan_error("slice assign", err);
        assert_eq!(mapped.identifier(), None);
        assert_eq!(mapped.message(), "slice assign: plain error");
    }

    #[test]
    fn typed_integer_scalar_range_and_symbolic_paths_ignore_f64_mirrors() {
        macro_rules! assert_typed_scalar {
            ($storage:expr, $expected:expr) => {{
                let tensor =
                    Tensor::new_integer($storage, vec![1, 1]).expect("typed integer scalar");
                let value = Value::Tensor(tensor);

                assert_eq!(
                    symbolic_scalar_from_value(&value).unwrap(),
                    SymbolicExpr::constant($expected)
                );
            }};
        }

        assert_typed_scalar!(IntegerStorage::I8(vec![-8]), -8.0);
        assert_typed_scalar!(IntegerStorage::I16(vec![-16]), -16.0);
        assert_typed_scalar!(IntegerStorage::I32(vec![-32]), -32.0);
        assert_typed_scalar!(IntegerStorage::I64(vec![-64]), -64.0);
        assert_typed_scalar!(IntegerStorage::U8(vec![8]), 8.0);
        assert_typed_scalar!(IntegerStorage::U16(vec![16]), 16.0);
        assert_typed_scalar!(IntegerStorage::U32(vec![32]), 32.0);
        assert_typed_scalar!(IntegerStorage::U64(vec![64]), 64.0);
    }
    #[test]
    fn resolve_cell_indices_rejects_fractional_values() {
        let err = block_on(super::resolve_cell_indices(&[Value::Num(1.5)]))
            .expect_err("fractional cell index should fail");
        assert_eq!(err.identifier(), Some("RunMat:CellIndexType"));
    }

    #[test]
    fn resolve_cell_indices_rejects_zero_values() {
        let err = block_on(super::resolve_cell_indices(&[Value::Num(0.0)]))
            .expect_err("zero cell index should fail");
        assert_eq!(err.identifier(), Some("RunMat:CellIndexOutOfBounds"));
    }

    #[test]
    fn resolve_cell_indices_accepts_scalar_tensor_values() {
        let scalar =
            Value::Tensor(runmat_value::Tensor::new(vec![2.0], vec![1, 1]).expect("scalar tensor"));
        let indices = block_on(super::resolve_cell_indices(&[scalar]))
            .expect("scalar tensor index should pass");
        assert_eq!(indices, vec![2]);
    }

    #[test]
    fn object_array_scalar_index_returns_object_and_slice_preserves_array() {
        let values = ["first", "second"]
            .into_iter()
            .map(|name| {
                let mut object = ObjectInstance::new("matlab.unittest.TestResult");
                object
                    .properties
                    .insert("Name".into(), Value::String(name.into()));
                Value::Object(object)
            })
            .collect();
        let array =
            Value::ObjectArray(ObjectArray::row("matlab.unittest.TestResult", values).unwrap());
        let registry = crate::bytecode::FunctionRegistry::default();
        let runtime = runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));

        let scalar = block_on(super::paren_index_value(
            &runtime,
            array.clone(),
            vec![Value::Num(2.0)],
            1,
            &registry,
        ))
        .unwrap();
        assert!(matches!(
            scalar,
            Value::Object(object)
                if object.properties.get("Name") == Some(&Value::String("second".into()))
        ));

        let selector =
            Value::Tensor(runmat_value::Tensor::new(vec![2.0, 1.0], vec![1, 2]).unwrap());
        let slice = block_on(super::paren_index_value(
            &runtime,
            array,
            vec![selector],
            1,
            &registry,
        ))
        .unwrap();
        assert!(matches!(
            slice,
            Value::ObjectArray(array)
                if array.shape() == [1, 2]
                    && array
                        .class_name()
                        .is(runmat_types::StaticClassIdentity::new("matlab.unittest.TestResult"))
        ));
    }

    #[test]
    fn logical_value_from_integer_tensor_uses_native_storage_not_mirror() {
        let tensor = Tensor::new_integer(IntegerStorage::U64(vec![0, u64::MAX]), vec![1, 2])
            .expect("typed integer tensor");

        let value = super::logical_value_from_tensor(tensor).expect("logical conversion");
        assert_eq!(
            value,
            Value::LogicalArray(
                runmat_value::LogicalArray::new(vec![0, 1], vec![1, 2]).expect("logical array")
            )
        );
    }
}
