//! Read the actual immutable public property, then use ordinary nested indexing.
use super::*;
use crate::object::indexing::{ObjectIndexSelector, ObjectSubscriptPath};
pub(super) async fn read(
    data: &ActiveSetWarmStartData,
    path: ObjectSubscriptPath,
    c: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    let (step, remaining) = path
        .steps()
        .split_first()
        .ok_or_else(|| policy().invalid("warm-start property path must not be empty"))?;
    let ObjectIndexSelector::Member(name) = step.selector() else {
        return Err(policy().invalid("a scalar warm start is read through its properties"));
    };
    let property =
        Property::parse(name).ok_or_else(|| policy().invalid("unknown warm-start property"))?;
    let value = match property {
        Property::X => data.x().clone(),
        Property::Options => data.options().clone(),
        Property::MaxLinearEqualities => {
            Value::Num(data.limits().equalities.map_or(f64::INFINITY, |v| v as f64))
        }
        Property::MaxLinearInequalities => Value::Num(
            data.limits()
                .inequalities
                .map_or(f64::INFINITY, |v| v as f64),
        ),
    };
    let value = RootedValue::in_optional_context(value, c.runtime())?;
    if remaining.is_empty() {
        return Ok(value);
    }
    let tail = ObjectSubscriptPath::new(remaining.to_vec())?;
    let runtime = c.runtime().cloned();
    value
        .with_owned_async(|value| async move {
            crate::object::protocol::read_subscript_path_with_runtime(runtime, value, tail, None)
                .await
        })
        .await
}
