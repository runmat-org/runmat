use crate::execution::RootedValue;
use runmat_value::Value;

use super::{AssignmentStepSpec, PreparedAssignmentStep, ReceiverCreationOrigin};
use crate::indexing::plan::IndexPlan;
use crate::object::dispatch::{
    invoke_resolved_object_index_path_method, resolve_object_index_protocol_with_access,
};
use crate::object::indexing::{
    ObjectIndexOp, ObjectIndexSelector, ObjectSubscript, ObjectSubscriptPath,
};
use crate::runtime_error::semantic_error;
use crate::RuntimeError;

mod writeback;
pub(super) use writeback::{assign_path, Writeback};

pub(super) async fn prepare_step_with_initialization(
    context: Option<&crate::context::RuntimeContext>,
    base: &Value,
    spec: AssignmentStepSpec,
    intent: runmat_types::AssignmentChildCreation,
    origin: ReceiverCreationOrigin,
) -> Result<PreparedAssignmentStep, RuntimeError> {
    let initialize_members = intent != runmat_types::AssignmentChildCreation::ExistingOnly;
    match spec {
        AssignmentStepSpec::Member(member) if initialize_members => {
            Ok(PreparedAssignmentStep::InitializingMember {
                member,
                intent,
                origin,
            })
        }
        AssignmentStepSpec::Member(member) => Ok(PreparedAssignmentStep::Member(member)),
        AssignmentStepSpec::Braces(indices) => match (base, initialize_members) {
            (Value::Cell(cell), true) => {
                let positions =
                    crate::object::cell::resolve_cell_assignment_child_components_in_context(
                        context, cell, &indices,
                    )?;
                Ok(PreparedAssignmentStep::InitializingCellContents { positions, intent })
            }
            _ => Ok(PreparedAssignmentStep::Braces(indices)),
        },
        AssignmentStepSpec::Parentheses { selectors } => {
            let shape = crate::indexing::value::shape_borrowed(base).ok_or_else(|| {
                semantic_error(
                    "CommaSeparatedListDestination",
                    "prepared destination path has no aggregate shape",
                )
            })?;
            let initializing =
                initialize_members && matches!(base, Value::Struct(_) | Value::StructArray(_));
            let plan = crate::object::indexing::plan_for_components_with_context(
                context,
                shape.as_slice(),
                &selectors,
                initializing,
            )
            .await?;
            if initializing {
                return Ok(PreparedAssignmentStep::InitializingParentheses {
                    plan,
                    selector_values: selectors,
                });
            }
            Ok(PreparedAssignmentStep::Parentheses {
                plan,
                selector_values: selectors,
            })
        }
    }
}

pub(super) fn selector_extent(
    base: &Value,
    component_count: usize,
    component: usize,
) -> Result<usize, RuntimeError> {
    let shape = crate::indexing::value::shape_borrowed(base).ok_or_else(|| {
        semantic_error(
            "CommaSeparatedListDestination",
            "prepared destination path has no aggregate shape",
        )
    })?;
    if component_count == 0 {
        return Err(semantic_error(
            "MissingIndex",
            "At least one index is required",
        ));
    }
    // The old shape constructor validated every extent before the component
    // lookup. Retain that validation/error ordering without allocating it.
    for dimension in 0..component_count {
        crate::indexing::plan::effective_index_extent(
            shape.as_slice(),
            component_count,
            dimension,
        )?;
    }
    if component >= component_count {
        return Err(semantic_error(
            "InvalidEndSelectorPlan",
            "selector component is outside the indexed rank",
        ));
    }
    crate::indexing::plan::effective_index_extent(shape.as_slice(), component_count, component)
}

pub(super) async fn read_step(
    context: Option<&crate::context::RuntimeContext>,
    base: Value,
    step: &PreparedAssignmentStep,
    caller: Option<&str>,
    access: Option<&crate::object::protocol::ObjectAccessContext>,
) -> Result<RootedValue, RuntimeError> {
    let owner = crate::indexing::plan::admission::Owner::new(context)?;
    let result = match step {
        PreparedAssignmentStep::InitializingMember {
            member,
            intent,
            origin,
        } => {
            let missing = match &base {
                Value::Struct(value) => {
                    *origin == ReceiverCreationOrigin::NewStructureElement
                        || !value.fields.contains_key(member.as_str())
                }
                Value::StructArray(value) if value.len() == 1 => {
                    *origin == ReceiverCreationOrigin::NewStructureElement
                        || value.column_owner(member.as_str()).is_none()
                }
                _ => false,
            };
            if missing {
                super::assignment_container_in_context(context, *intent)
            } else {
                crate::object::resolve::read_member_sequence_text_with_optional_access(
                    context,
                    base,
                    member.clone(),
                    true,
                    caller,
                    access,
                )
                .await?
                .try_into_single()
            }
        }
        PreparedAssignmentStep::Member(member) => {
            crate::object::resolve::read_member_sequence_text_with_optional_access(
                context,
                base,
                member.clone(),
                false,
                caller,
                access,
            )
            .await?
            .try_into_single()
        }
        PreparedAssignmentStep::Parentheses {
            plan,
            selector_values,
        } => read_parentheses(context, base, plan, selector_values, access).await,
        PreparedAssignmentStep::InitializingParentheses { plan, .. } => {
            crate::aggregate::structure::operations::project_initializing_in_context(
                context, &base, plan,
            )
        }
        PreparedAssignmentStep::InitializingCellContents { positions, intent } => {
            let Value::Cell(cell) = &base else {
                return Err(semantic_error(
                    "CellIndexType",
                    "Brace assignment requires a cell array",
                ));
            };
            crate::object::cell::read_cell_assignment_child_in_context(
                context, cell, positions, *intent,
            )
        }
        PreparedAssignmentStep::Braces(indices) => {
            match base {
                Value::Cell(cell) => {
                    let protocol = crate::object::indexing::array_in_context(
                        context,
                        indices.len(),
                        |index| indices[index].protocol_value_in_context(context),
                    )?;
                    crate::object::cell::read_cell_brace_single_in_context(
                        context, &cell, &protocol,
                    )
                }
                base @ (Value::Object(_) | Value::HandleObject(_)) => {
                    let path = ObjectSubscriptPath::single_in_context(
                        context,
                        ObjectSubscript::braces(ObjectIndexSelector::IndexValues {
                            components: indices.clone(),
                        }),
                    )?;
                    let resolution = resolve_object_index_protocol_with_access(
                        &base,
                        ObjectIndexOp::Subsref,
                        None,
                        access,
                    )?;
                    let values = match context {
                        Some(runtime) => crate::object::dispatch::invoke_resolved_object_index_path_method_in_context(runtime, &resolution, base, path, 1).await?,
                        None => invoke_resolved_object_index_path_method(&resolution, base, path, 1).await?,
                    };
                    // Original brace expansion requests one output, then pads
                    // or truncates to one while its source roots remain live.
                    let mut values = values.try_into_values()?;
                    RootedValue::in_optional_context(
                        values.next().unwrap_or(Value::Num(0.0)),
                        context,
                    )
                }
                _ => Err(semantic_error(
                    "CellExpansionOnNonCell",
                    "Cell expansion on non-cell",
                )),
            }
        }
    };
    let result = result?;
    owner.check()?;
    Ok(result)
}

pub(super) async fn write_step(
    context: Option<&crate::context::RuntimeContext>,
    base: Value,
    step: PreparedAssignmentStep,
    rhs: Value,
    caller: Option<&str>,
    access: Option<&crate::object::protocol::ObjectAccessContext>,
) -> Result<RootedValue, RuntimeError> {
    let owner = crate::indexing::plan::admission::Owner::new(context)?;
    let result = match step {
        PreparedAssignmentStep::Member(member)
        | PreparedAssignmentStep::InitializingMember { member, .. } => {
            crate::object::resolve::store_member_text_with_optional_access(
                context,
                base,
                member,
                rhs,
                caller,
                access,
                runmat_gc::gc_record_write,
            )
            .await
        }
        PreparedAssignmentStep::Parentheses {
            plan,
            selector_values,
        } => write_parentheses(context, base, &plan, selector_values, rhs, access).await,
        PreparedAssignmentStep::InitializingParentheses { plan, .. } => {
            crate::aggregate::structure::operations::assign_initializing_in_context(
                context, &base, &rhs, &plan,
            )
        }
        PreparedAssignmentStep::InitializingCellContents { positions, .. } => {
            let Value::Cell(cell) = base else {
                return Err(semantic_error(
                    "CellIndexType",
                    "Brace assignment requires a cell array",
                ));
            };
            crate::object::cell::assign_cell_value_in_context(
                context,
                cell,
                &positions,
                rhs,
                runmat_gc::gc_record_write,
            )
            .and_then(|value| RootedValue::in_optional_context(value, context))
        }
        PreparedAssignmentStep::Braces(indices) => {
            crate::sequence::destination::endpoint::assign_cell_contents_with_access(
                context,
                base,
                indices,
                crate::sequence::single_value_sequence(rhs)?,
                access,
            )
            .await
        }
    };
    let result = result?;
    owner.check()?;
    Ok(result)
}

async fn read_parentheses(
    context: Option<&crate::context::RuntimeContext>,
    base: Value,
    plan: &IndexPlan,
    selector_values: &runmat_value::ExactArrayStorage<
        crate::object::indexing::ObjectIndexComponent,
    >,
    access: Option<&crate::object::protocol::ObjectAccessContext>,
) -> Result<RootedValue, RuntimeError> {
    let resolution =
        resolve_object_index_protocol_with_access(&base, ObjectIndexOp::Subsref, None, access)?;
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
    crate::indexing::value::read_with_plan_in_context(context, base, plan)
        .and_then(|value| RootedValue::in_optional_context(value, context))
}

async fn write_parentheses(
    context: Option<&crate::context::RuntimeContext>,
    base: Value,
    plan: &IndexPlan,
    selector_values: runmat_value::ExactArrayStorage<crate::object::indexing::ObjectIndexComponent>,
    rhs: Value,
    access: Option<&crate::object::protocol::ObjectAccessContext>,
) -> Result<RootedValue, RuntimeError> {
    let resolution =
        resolve_object_index_protocol_with_access(&base, ObjectIndexOp::Subsasgn, None, access)?;
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
    crate::indexing::value::assign_with_plan_in_context(context, base, plan, rhs)
        .await
        .and_then(|value| RootedValue::in_optional_context(value, context))
}
