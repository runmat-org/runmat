use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;

use runmat_value::Value;

use super::{AssignmentStepSpec, PreparedAssignmentStep};
use crate::indexing::plan::IndexPlan;
use crate::object::dispatch::{
    invoke_resolved_object_index_path_method, resolve_object_index_protocol,
};
use crate::object::indexing::{
    ObjectIndexOp, ObjectIndexSelector, ObjectSubscript, ObjectSubscriptPath,
};
use crate::runtime_error::semantic_error;
use crate::sequence::destination::PreparedSequenceEndpoint;
use crate::RuntimeError;

pub(super) async fn prepare_step(
    base: &Value,
    spec: AssignmentStepSpec,
) -> Result<PreparedAssignmentStep, RuntimeError> {
    match spec {
        AssignmentStepSpec::Member(member) => Ok(PreparedAssignmentStep::Member(member)),
        AssignmentStepSpec::Braces(indices) => Ok(PreparedAssignmentStep::Braces(indices)),
        AssignmentStepSpec::Parentheses { selectors } => {
            let shape = aggregate_shape(base)?;
            let planned =
                crate::indexing::selectors::build_component_selectors(&selectors, &shape, false)
                    .await?;
            let plan = crate::indexing::plan::build_index_plan(&planned, selectors.len(), &shape)?;
            Ok(PreparedAssignmentStep::Parentheses {
                plan,
                selector_values: selectors,
            })
        }
    }
}

pub(super) fn aggregate_shape(base: &Value) -> Result<Vec<usize>, RuntimeError> {
    crate::indexing::value::shape(base).ok_or_else(|| {
        semantic_error(
            "CommaSeparatedListDestination",
            "prepared destination path has no aggregate shape",
        )
    })
}

pub(super) fn selector_extent(
    base: &Value,
    component_count: usize,
    component: usize,
) -> Result<usize, RuntimeError> {
    let shape = aggregate_shape(base)?;
    crate::indexing::plan::effective_index_shape(&shape, component_count)?
        .get(component)
        .copied()
        .ok_or_else(|| {
            semantic_error(
                "InvalidEndSelectorPlan",
                "selector component is outside the indexed rank",
            )
        })
}

pub(super) fn assign_path<'a>(
    current: Value,
    mut steps: VecDeque<PreparedAssignmentStep>,
    endpoint: PreparedSequenceEndpoint,
    values: Vec<Value>,
    caller: Option<&'a str>,
) -> Pin<Box<dyn Future<Output = Result<Value, RuntimeError>> + 'a>> {
    Box::pin(async move {
        if steps.is_empty() {
            return endpoint.assign(current, values, caller).await;
        }
        let step = steps.pop_front().expect("nonempty path checked above");
        let child = read_step(current.clone(), &step, caller).await?;
        let updated = assign_path(child, steps, endpoint, values, caller).await?;
        write_step(current, step, updated, caller).await
    })
}

pub(super) async fn read_step(
    base: Value,
    step: &PreparedAssignmentStep,
    caller: Option<&str>,
) -> Result<Value, RuntimeError> {
    match step {
        PreparedAssignmentStep::Member(member) => {
            crate::object::resolve::load_member(base, member.0.clone(), false, caller).await
        }
        PreparedAssignmentStep::Parentheses {
            plan,
            selector_values,
        } => read_parentheses(base, plan, selector_values).await,
        PreparedAssignmentStep::Braces(indices) => {
            let values = crate::call::arguments::expand_brace_values(
                base,
                &indices
                    .iter()
                    .map(|component| component.protocol_value())
                    .collect::<Vec<_>>(),
                Some(1),
            )
            .await?;
            match values.as_slice() {
                [value] => Ok(value.clone()),
                _ => Err(semantic_error(
                    "CommaSeparatedListRequiresSingleValue",
                    "a chained brace destination must select exactly one value",
                )),
            }
        }
    }
}

pub(super) async fn write_step(
    base: Value,
    step: PreparedAssignmentStep,
    rhs: Value,
    caller: Option<&str>,
) -> Result<Value, RuntimeError> {
    match step {
        PreparedAssignmentStep::Member(member) => {
            crate::object::resolve::store_member_traced(base, member.0, rhs, false, caller).await
        }
        PreparedAssignmentStep::Parentheses {
            plan,
            selector_values,
        } => write_parentheses(base, &plan, selector_values, rhs).await,
        PreparedAssignmentStep::Braces(indices) => {
            crate::sequence::destination::endpoint::assign_cell_contents(base, indices, vec![rhs])
                .await
        }
    }
}

async fn read_parentheses(
    base: Value,
    plan: &IndexPlan,
    selector_values: &[crate::object::indexing::ObjectIndexComponent],
) -> Result<Value, RuntimeError> {
    let resolution = resolve_object_index_protocol(&base, ObjectIndexOp::Subsref, None)?;
    if matches!(
        resolution,
        crate::object::protocol::ProtocolResolution::Method(_)
    ) {
        return invoke_resolved_object_index_path_method(
            &resolution,
            base,
            ObjectSubscriptPath::single(ObjectSubscript::parentheses(
                ObjectIndexSelector::IndexValues {
                    components: selector_values.to_vec(),
                },
            )),
            1,
        )
        .await;
    }
    crate::indexing::value::read_with_plan(base, plan)
}

async fn write_parentheses(
    base: Value,
    plan: &IndexPlan,
    selector_values: Vec<crate::object::indexing::ObjectIndexComponent>,
    rhs: Value,
) -> Result<Value, RuntimeError> {
    let resolution = resolve_object_index_protocol(&base, ObjectIndexOp::Subsasgn, None)?;
    if let crate::object::protocol::ProtocolResolution::Method(method) = resolution {
        return crate::object::protocol::invoke_resolved_object_assignment(
            &method,
            base,
            ObjectSubscriptPath::single(ObjectSubscript::parentheses(
                ObjectIndexSelector::IndexValues {
                    components: selector_values,
                },
            )),
            vec![rhs],
        )
        .await;
    }
    crate::indexing::value::assign_with_plan(base, plan, rhs).await
}
