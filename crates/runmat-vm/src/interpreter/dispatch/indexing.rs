use crate::interpreter::vars::InterpreterVars;
use runmat_runtime::call::arguments::expand_brace_values;
use runmat_runtime::execution::{RootedValue, RootedValueSequence};
use runmat_runtime::indexing::plan::build_index_plan;
use runmat_runtime::indexing::read_linear as idx_read_linear;
use runmat_runtime::indexing::read_slice as idx_read_slice;
use runmat_runtime::indexing::selectors::{
    build_scalar_selectors, build_slice_selectors, index_scalar_from_value, SliceSelector,
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
use runmat_value::{CellArray, Value};

mod scalar_read_and_cell;
mod scalar_store;
mod dispatch;
pub use dispatch::dispatch_indexing;

/// Admit the exact selector window while every source operand is still on
/// the stack. The separate root lease follows moved values through awaits.
fn extract_selector_operands(
    stack: &mut crate::interpreter::stack::InterpreterStack,
    runtime: &runmat_runtime::context::RuntimeContext,
    count: usize,
    has_rhs: bool,
) -> Result<
    (
        runmat_value::OutputListStorage,
        runmat_runtime::execution::RootedValueLease,
    ),
    RuntimeError,
> {
    let trailing = usize::from(has_rhs);
    let width = count.checked_add(trailing).and_then(|n| n.checked_add(1));
    let start = width
        .and_then(|width| stack.len().checked_sub(width))
        .ok_or_else(|| crate::interpreter::errors::mex("StackUnderflow", "stack underflow"))?;
    let roots = runmat_runtime::execution::RootedValueLease::for_values_iterator_in_context(
        stack.as_slice()[start..].iter(),
        runtime,
    )?;
    let values =
        stack.try_extract_output_range_in_context(start + 1, stack.len() - trailing, runtime)?;
    Ok((values, roots))
}
mod slice_read;
mod slice_store;

fn map_slice_plan_error(context: &str, err: RuntimeError) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{context}: {}", err.message()));
    if let Some(identifier) = err.identifier() {
        builder = builder.with_identifier(identifier.to_string());
    }
    builder.build()
}

fn root_value(value: Value) -> Result<RootedValue, RuntimeError> {
    RootedValue::new(value).map_err(|error| error.into_runtime_error())
}

fn resolve_single_sequence(sequence: RootedValueSequence) -> Result<RootedValue, RuntimeError> {
    sequence.try_map_owned(
        |sequence| {
            let values = sequence.resolve(
                runmat_types::SequenceUse::RequireSingle,
                runmat_runtime::sequence::SequenceResolutionContext::default(),
            )?;
            Ok(runmat_runtime::sequence::into_first_output(values)?
                .expect("one required index output was resolved"))
        },
        |error| error.into_runtime_error(),
    )
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
    runtime: &runmat_runtime::context::RuntimeContext,
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
    let components = runmat_runtime::context::host_allocation::initialize_host_array(
        Some(runtime),
        dims,
        |dimension| {
            if colon_mask & (1_u32 << dimension) != 0 {
                Ok(ObjectIndexComponent::Colon)
            } else {
                let value = values.next().ok_or_else(|| {
                    crate::interpreter::errors::mex(
                        "MalformedObjectSelector",
                        "object selector is missing an indexed component",
                    )
                })?;
                Ok(ObjectIndexComponent::Value(value.clone()))
            }
        },
    )?;
    if values.next().is_some() {
        return Err(crate::interpreter::errors::mex(
            "MalformedObjectSelector",
            "object selector contains excess indexed components",
        ));
    }
    Ok(ObjectIndexSelector::IndexValues { components })
}

fn object_index_components(
    runtime: &runmat_runtime::context::RuntimeContext,
    indices: Vec<Value>,
) -> Result<runmat_value::ExactArrayStorage<ObjectIndexComponent>, RuntimeError> {
    let count = indices.len();
    let mut indices = indices.into_iter();
    runmat_runtime::context::host_allocation::initialize_host_array(Some(runtime), count, |_| {
        Ok(indices.next().expect("exact selector count").into())
    })
}

async fn read_scalar_like_slice(
    runtime: &runmat_runtime::context::RuntimeContext,
    base: &Value,
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    numeric: &[Value],
) -> Result<Value, RuntimeError> {
    // Primitive scalars have a canonical 1-by-1 indexed shape, including
    // trailing singleton dimensions. The shared plan validates all selectors
    // and preserves empty and repeated selections.
    if dims > 0
        && matches!(
            base,
            Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::String(_)
        )
    {
        let selectors = runmat_runtime::indexing::selectors::build_slice_selectors_in_context(
            runtime,
            dims,
            colon_mask,
            end_mask,
            numeric,
            &[1usize, 1usize],
        )
        .await?;
        let plan = runmat_runtime::indexing::plan::build_index_plan_in_context(
            runtime,
            &selectors,
            dims,
            &[1usize, 1usize],
        )?;
        if plan.indices.as_slice() == [0] {
            return Ok(base.clone());
        }
        return runmat_runtime::indexing::value::read_with_plan_in_context(
            Some(runtime),
            base.clone(),
            &plan,
        );
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
    let selectors = runmat_runtime::indexing::selectors::build_slice_selectors_in_context(
        runtime,
        1,
        colon_mask,
        end_mask,
        numeric,
        &[1usize],
    )
    .await?;
    let count = match selectors.first() {
        Some(SliceSelector::Scalar(_)) => 1,
        Some(SliceSelector::Indices(indices)) => indices.len(),
        Some(SliceSelector::LinearIndices { values, .. }) => values.len(),
        Some(SliceSelector::Colon) | None => {
            return Err(crate::interpreter::errors::mex(
                "SliceNonTensor",
                "Slicing only supported on tensors",
            ));
        }
    };
    let linear_indices = runmat_runtime::context::host_allocation::initialize_host_array(
        Some(runtime),
        count,
        |ordinal| {
            Ok(match selectors.first() {
                Some(SliceSelector::Scalar(index)) => *index as f64,
                Some(SliceSelector::Indices(indices)) => indices[ordinal] as f64,
                Some(SliceSelector::LinearIndices { values, .. }) => values[ordinal] as f64,
                _ => unreachable!("selector kind checked before exact metadata construction"),
            })
        },
    )?;
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
        Ok(Some(_)) => return None,
        Err(error) => return Some(error),
        Ok(None) => {}
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
    indices: &[usize],
    rhs: Value,
    delete: bool,
) -> Result<RootedValue, RuntimeError> {
    let _roots = runmat_runtime::execution::RootedValueLease::for_values_iterator_in_context(
        [&base, &rhs].into_iter(),
        runtime,
    )?;
    let resolution = resolve_object_index_protocol(&base, ObjectIndexOp::Subsasgn, None)?;
    if let Some(method) = resolution {
        let components = runmat_runtime::context::host_allocation::initialize_host_array(
            Some(runtime),
            indices.len(),
            |ordinal| {
                Ok(ObjectIndexComponent::Value(Value::Num(
                    indices[ordinal] as f64,
                )))
            },
        )?;
        let path = ObjectSubscriptPath::single_in_context(
            Some(runtime),
            ObjectSubscript::parentheses(ObjectIndexSelector::IndexValues { components }),
        )?;
        return runmat_runtime::object::protocol::invoke_resolved_object_assignment_sequence_in_context(
            runtime,
            &method,
            base,
            path,
            runmat_runtime::call::arguments::adapt_legacy_builtin_result(rhs)?,
        )
        .await;
    }
    if delete {
        RootedValue::in_context(
            runmat_runtime::indexing::object::delete_scalar_indices_in_context(
                Some(runtime),
                base,
                indices,
                &rhs,
            )?,
            runtime,
        )
    } else {
        RootedValue::in_context(
            runmat_runtime::indexing::object::assign_scalar_indices_in_context(
                Some(runtime),
                base,
                indices,
                rhs,
            )?,
            runtime,
        )
    }
}

async fn invoke_object_brace_read(
    runtime: &runmat_runtime::context::RuntimeContext,
    base: Value,
    indices: Vec<Value>,
    requested_outputs: usize,
) -> Result<RootedValue, RuntimeError> {
    let _roots = crate::call::root_values(indices.iter().chain(std::iter::once(&base)))?;
    let resolution = resolve_object_index_protocol(&base, ObjectIndexOp::Subsref, None)?;
    let path = ObjectSubscriptPath::single_in_context(
        Some(runtime),
        ObjectSubscript::braces(ObjectIndexSelector::IndexValues {
            components: object_index_components(runtime, indices)?,
        }),
    )?;
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
) -> Result<RootedValue, RuntimeError> {
    let _roots = crate::call::root_values(indices.iter().chain([&base, &rhs]))?;
    let resolution = resolve_object_index_protocol(&base, ObjectIndexOp::Subsasgn, None)?;
    let Some(method) = resolution else {
        return Err(crate::interpreter::errors::mex(
            "MissingSubsasgn",
            "class does not define subsasgn for brace assignment",
        ));
    };
    let path = ObjectSubscriptPath::single_in_context(
        Some(runtime),
        ObjectSubscript::braces(ObjectIndexSelector::IndexValues {
            components: object_index_components(runtime, indices)?,
        }),
    )?;
    runmat_runtime::object::protocol::invoke_resolved_object_assignment_in_context(
        runtime,
        &method,
        base,
        path,
        runmat_runtime::sequence::into_unbounded_values(
            runmat_runtime::call::arguments::adapt_legacy_builtin_result(rhs)?,
        )?,
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
    for selector in &selectors {
        match selector {
            SliceSelector::Scalar(index) => indices.push(*index),
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

fn pop_index_values(
    stack: &mut crate::interpreter::stack::InterpreterStack,
    count: usize,
) -> Result<Vec<Value>, RuntimeError> {
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

fn pop_index_base(
    stack: &mut crate::interpreter::stack::InterpreterStack,
) -> Result<Value, RuntimeError> {
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
    Value(RootedValue),
    Expanded(runmat_runtime::execution::RootedValues),
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
            | Value::BoundFunctionHandle(_)
            | Value::Closure(_)
            | Value::ScopedFunctionHandle(_)
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
                    root_value(runmat_runtime::object::cell::index_cell_value(
                        &ca, &indices,
                    )?)?
                }
                Value::Composite(handle) => {
                    runmat_runtime::parallel::composite::read_composite_entry(
                        runtime,
                        handle,
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
                let value = runmat_runtime::parallel::composite::read_composite_entry(
                    runtime,
                    handle,
                    raw_indices,
                )
                .await?;
                return Ok(BraceIndexOutcome::Expanded({
                    let mut values =
                        runmat_runtime::execution::RootedValuesBuilder::with_capacity(1);
                    values.push_owned(value);
                    values
                        .finish()
                        .map_err(|error| error.into_runtime_error())?
                }));
            }
            let values = expand_brace_values(base, raw_indices, Some(out_count)).await?;
            Ok(BraceIndexOutcome::Expanded(values))
        }
        BraceIndexOperation::List => {
            if let Value::Composite(handle) = base {
                let value = runmat_runtime::parallel::composite::read_composite_entry(
                    runtime,
                    handle,
                    raw_indices,
                )
                .await?;
                return Ok(BraceIndexOutcome::Expanded({
                    let mut values =
                        runmat_runtime::execution::RootedValuesBuilder::with_capacity(1);
                    values.push_owned(value);
                    values
                        .finish()
                        .map_err(|error| error.into_runtime_error())?
                }));
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
                    let mut rhs_values = runmat_runtime::sequence::into_unbounded_values(rhs)?;
                    let updated = if raw_indices.len() > 1 {
                        let indices = resolve_cell_indices(raw_indices).await?;
                        match rhs_kind {
                            runmat_value::ValueSequenceKind::CommaSeparated
                                if rhs_values.len() == 1 => {
                                runmat_runtime::object::cell::assign_cell_value_in_context(
                                    Some(runtime),
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
                            runmat_value::ValueSequenceKind::Single => runmat_runtime::object::cell::assign_cell_value_in_context(
                                Some(runtime),
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
                            runmat_value::ValueSequenceKind::CommaSeparated => runmat_runtime::object::cell::assign_cell_value_multi_in_context(
                                Some(runtime),
                                ca,
                                &positions,
                                &rhs_values,
                                |oldv, newv| {
                                    runmat_gc::gc_record_write(oldv, newv);
                                },
                            )?,
                            runmat_value::ValueSequenceKind::Single if positions.len() == 1 => runmat_runtime::object::cell::assign_cell_value_in_context(
                                Some(runtime),
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
                    };
                    root_value(updated)?
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
    stack: &mut crate::interpreter::stack::InterpreterStack,
    _vars: &mut InterpreterVars,
    request: BraceStackRequest,
    runtime: &runmat_runtime::context::RuntimeContext,
) -> Result<(), RuntimeError> {
    let raw_indices = pop_index_values(stack, request.num_indices)?;
    let base = pop_index_base(stack)?;
    let outcome = execute_brace_operation(base, &raw_indices, request.operation, runtime).await?;
    match (outcome, request.expectation) {
        (BraceIndexOutcome::Value(value), BraceOutcomeExpectation::SingleValue { .. }) => {
            value.with_owned(|value| stack.try_push(value))?;
        }
        (BraceIndexOutcome::Expanded(values), BraceOutcomeExpectation::ExpandedValues { .. }) => {
            stack.try_extend(values.try_into_values()?)?;
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
) -> Result<RootedValueSequence, RuntimeError> {
    let _roots = crate::call::root_values(raw_indices.iter().chain(std::iter::once(&base)))?;
    if matches!(
        &base,
        Value::ObjectArray(_) | Value::Object(_) | Value::HandleObject(_)
    ) {
        let resolution = resolve_object_index_protocol(&base, ObjectIndexOp::Subsref, None)?;
        if resolution.is_some() {
            let path = ObjectSubscriptPath::single_in_context(
                Some(runtime),
                ObjectSubscript::parentheses(ObjectIndexSelector::IndexValues {
                    components: object_index_components(runtime, raw_indices)?,
                }),
            )?;
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
            | Value::BoundFunctionHandle(_)
            | Value::Closure(_)
            | Value::ScopedFunctionHandle(_)
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
    if let Value::Foreign(reference) = &base {
        let value =
            runmat_runtime::foreign::index_foreign_resource(reference.clone(), raw_indices).await?;
        return value.try_map_owned(
            |value| {
                runmat_value::ValueSequence::single(value)
                    .map_err(runmat_runtime::sequence::sequence_error_to_runtime)
            },
            |error| error.into_runtime_error(),
        );
    }
    let value = match &base {
        Value::Foreign(_) => unreachable!("foreign values returned above"),
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
        | Value::BoundFunctionHandle(_)
        | Value::Closure(_)
        | Value::ScopedFunctionHandle(_) => unreachable!("callable values returned above"),
        Value::Symbolic(expr) => {
            if let Some((name, parameters)) = expr.function_reference_signature() {
                runmat_runtime::indexing::symbolic::apply_function_reference(
                    name,
                    parameters,
                    &raw_indices,
                )
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
            if let Some(shape) = runmat_runtime::indexing::value::shape_borrowed(&base) {
                let selectors =
                    runmat_runtime::indexing::selectors::build_slice_selectors_in_context(
                        runtime,
                        raw_indices.len(),
                        0,
                        0,
                        &raw_indices,
                        shape.as_slice(),
                    )
                    .await?;
                let plan = runmat_runtime::indexing::plan::build_index_plan_in_context(
                    runtime,
                    &selectors,
                    raw_indices.len(),
                    shape.as_slice(),
                )?;
                runmat_runtime::indexing::value::read_with_plan_in_context(
                    Some(runtime),
                    base,
                    &plan,
                )
            } else {
                let indices = linear_index_values(&raw_indices).await?;
                idx_read_linear::generic_index(&base, &indices).await
            }
        }
    }?;
    let sequence = runmat_value::ValueSequence::single(value)
        .map_err(runmat_runtime::sequence::sequence_error_to_runtime)?;
    RootedValueSequence::new(sequence).map_err(|error| error.into_runtime_error())
}

pub async fn paren_index_value(
    runtime: &runmat_runtime::context::RuntimeContext,
    base: Value,
    raw_indices: Vec<Value>,
    requested_outputs: usize,
    function_registry: &crate::bytecode::FunctionRegistry,
) -> Result<RootedValue, RuntimeError> {
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

#[cfg(test)]
mod tests {
    use super::map_slice_plan_error;
    use futures::executor::block_on;
    use runmat_runtime::indexing::symbolic::scalar_from_value as symbolic_scalar_from_value;
    use runmat_value::{IntegerStorage, ObjectArray, SymbolicExpr, Tensor, Value};

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
                let mut object =
                    runmat_value::UnboundedObjectBuilder::new("matlab.unittest.TestResult");
                object.insert("Name".into(), Value::String(name.into()));
                Value::Object(object.finish())
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
            scalar.value(),
            Value::Object(object)
                if object.properties().get("Name") == Some(&Value::String("second".into()))
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
            slice.value(),
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

        let plan = runmat_runtime::indexing::plan::IndexPlan::new(
            vec![0, 1],
            vec![1, 2],
            vec![2],
            1,
            vec![1, 2],
        );
        let value =
            runmat_runtime::indexing::write_slice::assign_logical_with_plan_host_in_context(
                None,
                runmat_value::LogicalArray::new(vec![0, 0], vec![1, 2]).unwrap(),
                &plan,
                &Value::Tensor(tensor),
            )
            .expect("canonical typed logical assignment");
        assert_eq!(
            value,
            Value::LogicalArray(
                runmat_value::LogicalArray::new(vec![0, 1], vec![1, 2]).expect("logical array")
            )
        );
    }
}
