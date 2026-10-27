//! Select one typed property generation, then use the ordinary authenticated tail.
use super::*;
use crate::object::indexing::{ObjectIndexKind, ObjectIndexSelector, ObjectSubscriptPath};
pub(in crate::builtins::math::optim::optimmodel) async fn read(
    value: &Value,
    path: &ObjectSubscriptPath,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    let (property, rest) = select(path, policy)?;
    let value = get(value, property, policy, c).await?;
    if rest.is_empty() {
        return Ok(value);
    }
    super::super::protocol::read_ordinary(value, ObjectSubscriptPath::new(rest.to_vec())?, c).await
}
pub(in crate::builtins::math::optim::optimmodel) async fn assign(
    value: &Value,
    path: &ObjectSubscriptPath,
    rhs: &Value,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    let _inputs = policy.roots([value, rhs].into_iter(), c.runtime())?;
    let (property, rest) = select(path, policy)?;
    if property == Property::Variables {
        return Err(policy.invalid("Variables is a read-only property"));
    }
    if rest.is_empty() {
        return set(value, property, rhs, policy, c).await;
    }
    let mut current = get(value, property, policy, c).await?;
    if property == Property::Equations && rest[0].kind() == ObjectIndexKind::Member {
        let empty = match current.value() {
            Value::Tensor(value) => value.is_empty(),
            Value::StructArray(value) => value.is_empty() && value.field_names().len() == 0,
            _ => false,
        };
        if empty {
            current =
                crate::execution::structure::construct_struct_from_fields_in_optional_context(
                    c.runtime(),
                    std::iter::empty::<(&str, &Value)>(),
                )?;
        }
    }
    let next = super::super::protocol::assign_ordinary(
        current,
        ObjectSubscriptPath::new(rest.to_vec())?,
        rhs,
        c,
    )
    .await?;
    set(value, property, next.value(), policy, c).await
}
fn select(
    path: &ObjectSubscriptPath,
    policy: Policy,
) -> BuiltinResult<(Property, &[crate::object::indexing::ObjectSubscript])> {
    let (first, rest) = path
        .steps()
        .split_first()
        .ok_or_else(|| policy.invalid("equation problem property path is empty"))?;
    let ObjectIndexSelector::Member(name) = first.selector() else {
        return Err(policy.invalid("equation problem indexing requires a property name"));
    };
    let property =
        Property::parse(name).ok_or_else(|| policy.invalid("unknown equation problem property"))?;
    Ok((property, rest))
}
