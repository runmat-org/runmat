//! Expression subscripting retains the original graph lease through each
//! numerical/property operation. Immutable expression edits rebuild only the
//! selected path; ordinary property containers use the canonical object owner.
use super::*;
use crate::object::indexing::{
    ObjectIndexKind, ObjectIndexSelector, ObjectSubscript, ObjectSubscriptPath,
};
use runmat_builtins::OptimizationExpressionProperty as Property;

pub(super) async fn read(
    value: &Value,
    path: &ObjectSubscriptPath,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    let _receiver = owner::reference(value, policy)?;
    let mut base = RootedValue::in_optional_context(value.clone(), cooperation.runtime())?;
    for (position, step) in path.steps().iter().enumerate() {
        base = if matches!(base.value(), Value::Object(object) if is_expression(object)) {
            match step.kind() {
                ObjectIndexKind::Paren => {
                    indexing::read(base.value(), step, policy, cooperation).await?
                }
                ObjectIndexKind::Member => {
                    properties::get(base.value(), property(step, policy)?, policy, cooperation)
                        .await?
                }
                ObjectIndexKind::Brace => {
                    return Err(
                        policy.invalid("optimization expressions do not support brace indexing")
                    );
                }
            }
        } else {
            let remaining = ObjectSubscriptPath::new(path.steps()[position..].to_vec())?;
            return read_ordinary(base, remaining, cooperation).await;
        };
        cooperation.completed(1).await?;
    }
    Ok(base)
}

pub(super) async fn assign(
    value: &Value,
    path: &ObjectSubscriptPath,
    rhs: &Value,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    let _inputs = policy.roots([value, rhs].into_iter(), cooperation.runtime())?;
    let _receiver = owner::reference(value, policy)?;
    let mut base = RootedValue::in_optional_context(value.clone(), cooperation.runtime())?;
    let mut parents = Vec::new();
    let mut result = None;
    for (position, step) in path.steps().iter().enumerate() {
        match step.kind() {
            ObjectIndexKind::Paren if position + 1 < path.steps().len() => {
                let child = indexing::read(base.value(), step, policy, cooperation).await?;
                parents
                    .try_reserve(1)
                    .map_err(|source| policy.reserve(source))?;
                parents.push((base, step));
                base = child;
            }
            ObjectIndexKind::Paren => {
                result =
                    Some(indexing::assign(base.value(), step, rhs, policy, cooperation).await?);
                break;
            }
            ObjectIndexKind::Member => {
                let selected = property(step, policy)?;
                if selected == Property::Variables {
                    return Err(policy.invalid("Variables is a read-only property"));
                }
                let replacement = if position + 1 == path.steps().len() {
                    RootedValue::in_optional_context(rhs.clone(), cooperation.runtime())?
                } else {
                    let current =
                        properties::get(base.value(), selected, policy, cooperation).await?;
                    let remaining =
                        ObjectSubscriptPath::new(path.steps()[position + 1..].to_vec())?;
                    assign_ordinary(current, remaining, rhs, cooperation).await?
                };
                result = Some(
                    properties::set_names(base.value(), replacement.value(), policy, cooperation)
                        .await?,
                );
                break;
            }
            ObjectIndexKind::Brace => {
                return Err(
                    policy.invalid("optimization expressions do not support brace assignment")
                );
            }
        }
    }
    let mut result =
        result.ok_or_else(|| policy.invalid("expression assignment requires a nonempty path"))?;
    while let Some((parent, step)) = parents.pop() {
        result =
            indexing::assign(parent.value(), step, result.value(), policy, cooperation).await?;
    }
    Ok(result)
}

fn property(step: &ObjectSubscript, policy: Policy) -> BuiltinResult<Property> {
    let ObjectIndexSelector::Member(name) = step.selector() else {
        return Err(policy.invalid("expression property requires a member selector"));
    };
    Property::parse(name).ok_or_else(|| policy.invalid("unknown optimization-expression property"))
}

pub(super) async fn read_ordinary(
    base: RootedValue,
    path: ObjectSubscriptPath,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    let runtime = cooperation.runtime().cloned();
    base.with_owned_async(|value| async move {
        crate::object::protocol::read_subscript_path_with_runtime(runtime, value, path, None).await
    })
    .await
}

pub(super) async fn assign_ordinary(
    base: RootedValue,
    path: ObjectSubscriptPath,
    rhs: &Value,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    let runtime = cooperation.runtime().cloned();
    let supplied = vec![rhs.clone()];
    base.with_owned_async(|value| async move {
        crate::object::protocol::assign_subscript_path_with_runtime(
            runtime, value, path, supplied, None,
        )
        .await
    })
    .await
}
