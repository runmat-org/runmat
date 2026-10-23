use runmat_value::ExactArrayIntoIter;

use runmat_value::Value;

use crate::execution::{RootedValue, RootedValueSequence};
use crate::object::indexing::{
    ObjectIndexKind, ObjectIndexSelector, ObjectSubscript, ObjectSubscriptPath,
};
use crate::runtime_error::semantic_error;
use crate::sequence::ValueSequence;
use crate::RuntimeError;

mod errors;
use errors::{invalid_base, invalid_path, invalid_selector};

pub(super) async fn read_default_step_sequence(
    runtime: Option<&crate::context::RuntimeContext>,
    base: Value,
    step: &ObjectSubscript,
    caller: Option<&str>,
) -> Result<RootedValueSequence, RuntimeError> {
    match step.kind() {
        ObjectIndexKind::Member => {
            let ObjectIndexSelector::Member(member) = step.selector() else {
                return Err(invalid_selector());
            };
            crate::object::resolve::read_member_sequence_text_with_context(
                runtime,
                base,
                member.clone(),
                false,
                caller,
            )
            .await
        }
        ObjectIndexKind::Brace => read_brace_sequence(runtime, base, step).await,
        ObjectIndexKind::Paren => read_default_step(runtime, base, step, caller)
            .await?
            .try_map_owned(
                |value| {
                    ValueSequence::single(value).map_err(crate::sequence::sequence_error_to_runtime)
                },
                |error| error.into_runtime_error(),
            ),
    }
}

pub(super) fn assign_default_steps<'a>(
    runtime: Option<&'a crate::context::RuntimeContext>,
    base: Value,
    mut steps: ExactArrayIntoIter<ObjectSubscript>,
    values: Vec<Value>,
    caller: Option<&'a str>,
) -> crate::execution::RuntimeContinuationFuture<'a, RootedValue> {
    super::super::context::admit_future_with(runtime, move || async move {
        let _inputs_root = crate::execution::RootedValueLease::for_values(
            std::iter::once(&base).chain(values.iter()),
        )
        .map_err(|error| error.into_runtime_error())?;
        let step = steps.next().ok_or_else(invalid_path)?;
        if steps.len() == 0 {
            return write_default_step(runtime, base, step, values, caller).await;
        }
        let child = read_default_step(runtime, base.clone(), &step, caller).await?;
        let remaining = ObjectSubscriptPath::from_remaining_in_context(runtime, steps)?;
        let updated = match runtime {
            Some(runtime) => {
                super::assign_subscript_path_in_context(
                    runtime,
                    child.value().clone(),
                    remaining,
                    values,
                    caller,
                )
                .await?
            }
            None => {
                super::assign_subscript_path(child.value().clone(), remaining, values, caller)
                    .await?
            }
        };
        write_default_step(runtime, base, step, vec![updated.value().clone()], caller).await
    })
}

pub(super) async fn read_default_step(
    runtime: Option<&crate::context::RuntimeContext>,
    base: Value,
    step: &ObjectSubscript,
    caller: Option<&str>,
) -> Result<RootedValue, RuntimeError> {
    match step.kind() {
        ObjectIndexKind::Member => {
            let ObjectIndexSelector::Member(member) = step.selector() else {
                return Err(invalid_selector());
            };
            crate::object::resolve::read_member_sequence_text_with_context(
                runtime,
                base,
                member.clone(),
                false,
                caller,
            )
            .await?
            .try_into_single()
        }
        ObjectIndexKind::Paren => {
            let values = index_values(step)?;
            let plan = plan_for(runtime, &base, values, false).await?;
            RootedValue::in_optional_context(
                crate::indexing::value::read_with_plan_in_context(runtime, base, &plan)?,
                runtime,
            )
        }
        ObjectIndexKind::Brace => read_brace_sequence(runtime, base, step)
            .await?
            .try_into_single(),
    }
}

async fn read_brace_sequence(
    runtime: Option<&crate::context::RuntimeContext>,
    base: Value,
    step: &ObjectSubscript,
) -> Result<RootedValueSequence, RuntimeError> {
    let _base_root = crate::execution::RootedValueLease::for_values(std::iter::once(&base))
        .map_err(|error| error.into_runtime_error())?;
    let indices = protocol_values(index_values(step)?);
    match base {
        Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle(_)
        | Value::Closure(_) => Err(semantic_error(
            "UnsupportedFunctionHandleSelector",
            "Function handle call does not support brace selector syntax",
        )),
        Value::Composite(handle) => {
            let runtime = runtime.ok_or_else(|| {
                semantic_error(
                    "RuntimeCapabilityUnavailable",
                    "Composite brace indexing requires a selected runtime",
                )
            })?;
            crate::parallel::composite::read_composite_entry(runtime, handle, &indices)
                .await?
                .try_map_owned(
                    |value| {
                        ValueSequence::comma_separated(vec![value])
                            .map_err(crate::sequence::sequence_error_to_runtime)
                    },
                    |error| error.into_runtime_error(),
                )
        }
        base @ (Value::Cell(_) | Value::Object(_) | Value::HandleObject(_)) => {
            crate::call::arguments::expand_brace_values(base, &indices, None)
                .await?
                .with_owned_output_storage(|values| {
                    RootedValueSequence::new(
                        ValueSequence::from_output_storage(values)
                            .map_err(crate::sequence::sequence_error_to_runtime)?,
                    )
                    .map_err(|error| error.into_runtime_error())
                })
        }
        _ => Err(semantic_error(
            "CellIndexingOnNonCell",
            "Cell indexing on non-cell",
        )),
    }
}

async fn write_default_step(
    runtime: Option<&crate::context::RuntimeContext>,
    base: Value,
    step: ObjectSubscript,
    values: Vec<Value>,
    caller: Option<&str>,
) -> Result<RootedValue, RuntimeError> {
    let _inputs_root =
        crate::execution::RootedValueLease::for_values(std::iter::once(&base).chain(values.iter()))
            .map_err(|error| error.into_runtime_error())?;
    match step.kind() {
        ObjectIndexKind::Member => {
            let ObjectIndexSelector::Member(member) = step.selector() else {
                return Err(invalid_selector());
            };
            match runtime {
                Some(runtime) => {
                    crate::object::resolve::store_member_sequence_values_in_context(
                        runtime,
                        base,
                        member.clone(),
                        ValueSequence::comma_separated(values)
                            .map_err(crate::sequence::sequence_error_to_runtime)?,
                        caller,
                    )
                    .await
                }
                None => {
                    crate::object::resolve::store_member_sequence_values(
                        base,
                        member.clone(),
                        ValueSequence::comma_separated(values)
                            .map_err(crate::sequence::sequence_error_to_runtime)?,
                        caller,
                    )
                    .await
                }
            }
        }
        ObjectIndexKind::Paren => {
            let [rhs] = values.as_slice() else {
                return Err(crate::runtime_error::semantic_error(
                    "CommaSeparatedListAssignmentArity",
                    "parentheses assignment requires one value",
                ));
            };
            let plan = plan_for(runtime, &base, index_values(&step)?, true).await?;
            RootedValue::in_optional_context(
                crate::indexing::value::assign_with_plan_in_context(
                    runtime,
                    base,
                    &plan,
                    rhs.clone(),
                )
                .await?,
                runtime,
            )
        }
        ObjectIndexKind::Brace => {
            let (ObjectIndexSelector::IndexValues { components }, _) = step.into_parts() else {
                return Err(invalid_selector());
            };
            crate::sequence::destination::endpoint::assign_cell_contents_with_context(
                runtime,
                base,
                components,
                ValueSequence::comma_separated(values)
                    .map_err(crate::sequence::sequence_error_to_runtime)?,
            )
            .await
        }
    }
}

fn plan_for<'a>(
    runtime: Option<&'a crate::context::RuntimeContext>,
    base: &'a Value,
    values: &'a [crate::object::indexing::ObjectIndexComponent],
    assignment: bool,
) -> impl std::future::Future<Output = Result<crate::indexing::plan::IndexPlan, RuntimeError>> + 'a
{
    let Some(shape) = crate::indexing::value::shape_borrowed(base) else {
        return futures::future::Either::Left(async { Err(invalid_base()) });
    };
    // Transfer the existing borrowed/inline geometry into the original
    // planner. No shape clone or intermediate planning poll frame is needed.
    futures::future::Either::Right(crate::object::indexing::plan_for_components_with_context(
        runtime, shape, values, assignment,
    ))
}

fn index_values(
    step: &ObjectSubscript,
) -> Result<&[crate::object::indexing::ObjectIndexComponent], RuntimeError> {
    match step.selector() {
        ObjectIndexSelector::IndexValues { components } => Ok(components),
        _ => Err(invalid_selector()),
    }
}

fn protocol_values(components: &[crate::object::indexing::ObjectIndexComponent]) -> Vec<Value> {
    components
        .iter()
        .map(|value| value.protocol_value())
        .collect()
}
