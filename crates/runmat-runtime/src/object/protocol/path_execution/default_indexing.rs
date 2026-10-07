use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;

use runmat_value::Value;

use crate::object::indexing::{
    ObjectIndexKind, ObjectIndexSelector, ObjectSubscript, ObjectSubscriptPath,
};
use crate::runtime_error::semantic_error;
use crate::sequence::{ResolveValueSequence, SequenceResolutionContext, ValueSequence};
use crate::RuntimeError;

mod errors;
use errors::{invalid_base, invalid_path, invalid_selector, single_value_error};

pub(super) async fn read_default_step_sequence(
    runtime: Option<&crate::context::RuntimeContext>,
    base: Value,
    step: &ObjectSubscript,
    caller: Option<&str>,
) -> Result<ValueSequence, RuntimeError> {
    match step.kind() {
        ObjectIndexKind::Member => {
            let ObjectIndexSelector::Member(member) = step.selector() else {
                return Err(invalid_selector());
            };
            crate::object::resolve::read_member_sequence_with_context(
                runtime,
                base,
                member.0.clone(),
                false,
                caller,
            )
            .await
        }
        ObjectIndexKind::Brace => read_brace_sequence(runtime, base, step).await,
        ObjectIndexKind::Paren => read_default_step(runtime, base, step, caller)
            .await
            .and_then(|value| {
                ValueSequence::single(value).map_err(crate::sequence::sequence_error_to_runtime)
            }),
    }
}

pub(super) fn assign_default_steps<'a>(
    runtime: Option<&'a crate::context::RuntimeContext>,
    base: Value,
    mut steps: VecDeque<ObjectSubscript>,
    values: Vec<Value>,
    caller: Option<&'a str>,
) -> Pin<Box<dyn Future<Output = Result<Value, RuntimeError>> + 'a>> {
    Box::pin(async move {
        let step = steps.pop_front().ok_or_else(invalid_path)?;
        if steps.is_empty() {
            return write_default_step(runtime, base, step, values, caller).await;
        }
        let child = read_default_step(runtime, base.clone(), &step, caller).await?;
        let remaining = ObjectSubscriptPath::new(steps.into_iter().collect())?;
        let updated = match runtime {
            Some(runtime) => {
                super::assign_subscript_path_in_context(runtime, child, remaining, values, caller)
                    .await?
            }
            None => super::assign_subscript_path(child, remaining, values, caller).await?,
        };
        write_default_step(runtime, base, step, vec![updated], caller).await
    })
}

pub(super) async fn read_default_step(
    runtime: Option<&crate::context::RuntimeContext>,
    base: Value,
    step: &ObjectSubscript,
    caller: Option<&str>,
) -> Result<Value, RuntimeError> {
    match step.kind() {
        ObjectIndexKind::Member => {
            let ObjectIndexSelector::Member(member) = step.selector() else {
                return Err(invalid_selector());
            };
            let mut values = crate::object::resolve::read_member_sequence_with_context(
                runtime,
                base,
                member.0.clone(),
                false,
                caller,
            )
            .await?
            .resolve(
                runmat_types::SequenceUse::RequireSingle,
                SequenceResolutionContext::default(),
            )?;
            values.pop().ok_or_else(single_value_error)
        }
        ObjectIndexKind::Paren => {
            let values = index_values(step)?;
            let plan = plan_for(&base, values, false).await?;
            crate::indexing::value::read_with_plan(base, &plan)
        }
        ObjectIndexKind::Brace => {
            let mut values = read_brace_sequence(runtime, base, step).await?.resolve(
                runmat_types::SequenceUse::RequireSingle,
                SequenceResolutionContext::default(),
            )?;
            values.pop().ok_or_else(single_value_error)
        }
    }
}

async fn read_brace_sequence(
    runtime: Option<&crate::context::RuntimeContext>,
    base: Value,
    step: &ObjectSubscript,
) -> Result<ValueSequence, RuntimeError> {
    let indices = protocol_values(index_values(step)?);
    let values = match base {
        Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::Closure(_) => {
            return Err(semantic_error(
                "UnsupportedFunctionHandleSelector",
                "Function handle call does not support brace selector syntax",
            ));
        }
        Value::Composite(handle) => {
            let runtime = runtime.ok_or_else(|| {
                semantic_error(
                    "RuntimeCapabilityUnavailable",
                    "Composite brace indexing requires a selected runtime",
                )
            })?;
            vec![
                crate::parallel::composite::read_composite_entry(runtime, *handle, &indices)
                    .await?,
            ]
        }
        base @ (Value::Cell(_) | Value::Object(_) | Value::HandleObject(_)) => {
            crate::call::arguments::expand_brace_values(base, &indices, None).await?
        }
        _ => {
            return Err(semantic_error(
                "CellIndexingOnNonCell",
                "Cell indexing on non-cell",
            ));
        }
    };
    ValueSequence::comma_separated(values).map_err(crate::sequence::sequence_error_to_runtime)
}

async fn write_default_step(
    runtime: Option<&crate::context::RuntimeContext>,
    base: Value,
    step: ObjectSubscript,
    values: Vec<Value>,
    caller: Option<&str>,
) -> Result<Value, RuntimeError> {
    match step.kind() {
        ObjectIndexKind::Member => {
            let ObjectIndexSelector::Member(member) = step.selector() else {
                return Err(invalid_selector());
            };
            match runtime {
                Some(runtime) => {
                    crate::object::resolve::store_member_sequence_traced_in_context(
                        runtime,
                        base,
                        member.0.clone(),
                        values,
                        caller,
                    )
                    .await
                }
                None => {
                    crate::object::resolve::store_member_sequence_traced(
                        base,
                        member.0.clone(),
                        values,
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
            let plan = plan_for(&base, index_values(&step)?, true).await?;
            crate::indexing::value::assign_with_plan(base, &plan, rhs.clone()).await
        }
        ObjectIndexKind::Brace => {
            crate::sequence::destination::endpoint::assign_cell_contents_with_context(
                runtime,
                base,
                index_values(&step)?.to_vec(),
                values,
            )
            .await
        }
    }
}

async fn plan_for(
    base: &Value,
    values: &[crate::object::indexing::ObjectIndexComponent],
    assignment: bool,
) -> Result<crate::indexing::plan::IndexPlan, RuntimeError> {
    let shape = crate::indexing::value::shape(base).ok_or_else(invalid_base)?;
    crate::object::indexing::plan_for_components(&shape, values, assignment).await
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
