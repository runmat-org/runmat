//! One problem property generation is selected before delegating the exact
//! remaining path to the ordinary authenticated object protocol.
use super::*;
use crate::object::indexing::{ObjectIndexKind, ObjectIndexSelector, ObjectSubscriptPath};
pub(in crate::builtins::math::optim::optimmodel) async fn read(
    value: &Value,
    path: &ObjectSubscriptPath,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    let (property, remaining) = select(path, policy)?;
    let result = get(value, property, policy, cooperation).await?;
    if remaining.is_empty() {
        return Ok(result);
    }
    let remaining = ObjectSubscriptPath::new(remaining.to_vec())?;
    super::super::protocol::read_ordinary(result, remaining, cooperation).await
}
pub(in crate::builtins::math::optim::optimmodel) async fn assign(
    value: &Value,
    path: &ObjectSubscriptPath,
    rhs: &Value,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    let _inputs = policy.roots([value, rhs].into_iter(), cooperation.runtime())?;
    let (property, remaining) = select(path, policy)?;
    if property == Property::Variables {
        return Err(policy.invalid("Variables is a read-only property"));
    }
    if remaining.is_empty() {
        return set(value, property, rhs, policy, cooperation).await;
    }
    let current = get(value, property, policy, cooperation).await?;
    let current = initial_member_container(
        value,
        property,
        current,
        remaining[0].kind(),
        policy,
        cooperation,
    )
    .await?;
    let remaining = ObjectSubscriptPath::new(remaining.to_vec())?;
    let successor =
        super::super::protocol::assign_ordinary(current, remaining, rhs, cooperation).await?;
    set(value, property, successor.value(), policy, cooperation).await
}
fn select(
    path: &ObjectSubscriptPath,
    policy: Policy,
) -> BuiltinResult<(Property, &[crate::object::indexing::ObjectSubscript])> {
    let (first, rest) = path
        .steps()
        .split_first()
        .ok_or_else(|| policy.invalid("problem property path is empty"))?;
    let ObjectIndexSelector::Member(name) = first.selector() else {
        return Err(policy.invalid("optimization problem indexing requires a property name"));
    };
    let property = Property::parse(name)
        .ok_or_else(|| policy.invalid("unknown optimization problem property"))?;
    Ok((property, rest))
}
async fn initial_member_container(
    problem: &Value,
    property: Property,
    current: RootedValue,
    next: ObjectIndexKind,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    if next != ObjectIndexKind::Member {
        return Ok(current);
    }
    if property == Property::Objective {
        if let Value::Object(object) = current.value() {
            if is_expression(object) {
                let node =
                    owner::OwnedNode::read(owner::reference(current.value(), policy)?, policy)?;
                if node.node().shape.contains(&0) {
                    return RootedValue::in_optional_context(
                        Value::Struct(runmat_value::StructValue::new()),
                        cooperation.runtime(),
                    );
                }
            }
        }
    }
    if property == Property::Constraints
        && matches!(current.value(),Value::StructArray(array) if array.is_empty() && array.field_names().len()==0)
    {
        return RootedValue::in_optional_context(
            Value::Struct(runmat_value::StructValue::new()),
            cooperation.runtime(),
        );
    }
    if property == Property::ObjectiveSense {
        let model = data(problem, policy)?;
        if let Senses::Uniform(sense) = &model.sense {
            let objectives = if let Value::Struct(fields) = &model.objective {
                Some(fields)
            } else {
                None
            };
            let text = RootedValue::in_optional_context(
                Value::String(Utf8Text::new(sense.name().to_owned()).into()),
                cooperation.runtime(),
            )?;
            return crate::execution::structure::construct_struct_from_fields_in_optional_context(
                cooperation.runtime(),
                objectives
                    .into_iter()
                    .flat_map(|fields| fields.field_names())
                    .map(|name| (name, text.value())),
            );
        }
    }
    Ok(current)
}
