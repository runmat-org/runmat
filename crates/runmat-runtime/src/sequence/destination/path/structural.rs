use super::super::control::ControlIter;
use crate::execution::RuntimeContinuationFuture;

use crate::execution::RootedValue;
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

pub(super) async fn prepare_step_with_initialization(
    context: Option<&crate::context::RuntimeContext>,
    base: &Value,
    spec: AssignmentStepSpec,
    initialize_members: bool,
) -> Result<PreparedAssignmentStep, RuntimeError> {
    match spec {
        AssignmentStepSpec::Member(member) if initialize_members => {
            Ok(PreparedAssignmentStep::InitializingMember(member))
        }
        AssignmentStepSpec::Member(member) => Ok(PreparedAssignmentStep::Member(member)),
        AssignmentStepSpec::Braces(indices) => Ok(PreparedAssignmentStep::Braces(indices)),
        AssignmentStepSpec::Parentheses { selectors } => {
            let shape = crate::indexing::value::shape_borrowed(base).ok_or_else(|| {
                semantic_error(
                    "CommaSeparatedListDestination",
                    "prepared destination path has no aggregate shape",
                )
            })?;
            let plan = crate::object::indexing::plan_for_components_with_context(
                context,
                shape.as_slice(),
                &selectors,
                false,
            )
            .await?;
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
    context: Option<&'a crate::context::RuntimeContext>,
    current: Value,
    mut steps: ControlIter<PreparedAssignmentStep>,
    endpoint: PreparedSequenceEndpoint,
    values: runmat_value::ValueSequence,
    caller: Option<&'a str>,
) -> RuntimeContinuationFuture<'a, RootedValue> {
    super::super::control::admit_future(context, async move {
        if steps.len() == 0 {
            return match context {
                Some(context) => {
                    endpoint
                        .assign_sequence_in_context(context, current, values, caller)
                        .await
                }
                None => endpoint.assign_sequence(current, values, caller).await,
            };
        }
        let step = steps.next().expect("nonempty path checked above");
        let child = read_step(context, current.clone(), &step, caller).await?;
        let updated = child
            .with_owned_async(|child| assign_path(context, child, steps, endpoint, values, caller))
            .await?;
        updated
            .with_owned_async(|updated| write_step(context, current, step, updated, caller))
            .await
    })
}

pub(super) async fn read_step(
    context: Option<&crate::context::RuntimeContext>,
    base: Value,
    step: &PreparedAssignmentStep,
    caller: Option<&str>,
) -> Result<RootedValue, RuntimeError> {
    match step {
        PreparedAssignmentStep::Member(member)
        | PreparedAssignmentStep::InitializingMember(member) => {
            crate::object::resolve::read_member_sequence_text_with_context(
                context,
                base,
                member.clone(),
                matches!(step, PreparedAssignmentStep::InitializingMember(_)),
                caller,
            )
            .await?
            .try_into_single()
        }
        PreparedAssignmentStep::Parentheses {
            plan,
            selector_values,
        } => read_parentheses(context, base, plan, selector_values).await,
        PreparedAssignmentStep::Braces(indices) => {
            let protocol =
                crate::object::indexing::array_in_context(context, indices.len(), |index| {
                    indices[index].protocol_value_in_context(context)
                })?;
            let values =
                crate::call::arguments::expand_brace_values(base, &protocol, Some(1)).await?;
            let mut values = values.try_into_values()?;
            if values.len() != 1 {
                return Err(semantic_error(
                    "CommaSeparatedListRequiresSingleValue",
                    "a chained brace destination must select exactly one value",
                ));
            }
            RootedValue::in_optional_context(
                values.next().expect("one chained brace value"),
                context,
            )
        }
    }
}

pub(super) async fn write_step(
    context: Option<&crate::context::RuntimeContext>,
    base: Value,
    step: PreparedAssignmentStep,
    rhs: Value,
    caller: Option<&str>,
) -> Result<RootedValue, RuntimeError> {
    let initialize_members = matches!(&step, PreparedAssignmentStep::InitializingMember(_));
    match step {
        PreparedAssignmentStep::Member(member)
        | PreparedAssignmentStep::InitializingMember(member) => {
            crate::object::resolve::store_member_text_with_context(
                context,
                base,
                member,
                rhs,
                initialize_members,
                caller,
                runmat_gc::gc_record_write,
            )
            .await
        }
        PreparedAssignmentStep::Parentheses {
            plan,
            selector_values,
        } => write_parentheses(context, base, &plan, selector_values, rhs).await,
        PreparedAssignmentStep::Braces(indices) => {
            crate::sequence::destination::endpoint::assign_cell_contents_with_context(
                context,
                base,
                indices,
                crate::sequence::single_value_sequence(rhs)?,
            )
            .await
        }
    }
}

async fn read_parentheses(
    context: Option<&crate::context::RuntimeContext>,
    base: Value,
    plan: &IndexPlan,
    selector_values: &runmat_value::ExactArrayStorage<
        crate::object::indexing::ObjectIndexComponent,
    >,
) -> Result<RootedValue, RuntimeError> {
    let resolution = resolve_object_index_protocol(&base, ObjectIndexOp::Subsref, None)?;
    if resolution.is_some() {
        let path = ObjectSubscriptPath::single_in_context(
            context,
            ObjectSubscript::parentheses(ObjectIndexSelector::IndexValues {
                components: selector_values.clone(),
            }),
        )?;
        let sequence = match context {
            Some(context) => {
                crate::object::dispatch::invoke_resolved_object_index_path_method_in_context(
                    context,
                    &resolution,
                    base,
                    path,
                    1,
                )
                .await?
            }
            None => invoke_resolved_object_index_path_method(&resolution, base, path, 1).await?,
        };
        return sequence.try_into_single();
    }
    crate::indexing::value::read_with_plan(base, plan)
        .and_then(|value| RootedValue::new(value).map_err(|error| error.into_runtime_error()))
}

async fn write_parentheses(
    context: Option<&crate::context::RuntimeContext>,
    base: Value,
    plan: &IndexPlan,
    selector_values: runmat_value::ExactArrayStorage<crate::object::indexing::ObjectIndexComponent>,
    rhs: Value,
) -> Result<RootedValue, RuntimeError> {
    let resolution = resolve_object_index_protocol(&base, ObjectIndexOp::Subsasgn, None)?;
    if let Some(method) = resolution {
        let path = ObjectSubscriptPath::single_in_context(
            context,
            ObjectSubscript::parentheses(ObjectIndexSelector::IndexValues {
                components: selector_values,
            }),
        )?;
        return match context {
            Some(context) => {
                crate::object::protocol::invoke_resolved_object_assignment_sequence_in_context(
                    context,
                    &method,
                    base,
                    path,
                    crate::sequence::single_value_sequence(rhs)?,
                )
                .await
            }
            None => {
                crate::object::protocol::invoke_resolved_object_assignment_sequence(
                    &method,
                    base,
                    path,
                    crate::sequence::single_value_sequence(rhs)?,
                )
                .await
            }
        };
    }
    crate::indexing::value::assign_with_plan(base, plan, rhs)
        .await
        .and_then(|value| RootedValue::new(value).map_err(|error| error.into_runtime_error()))
}
