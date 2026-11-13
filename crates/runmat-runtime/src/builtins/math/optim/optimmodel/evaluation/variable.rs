//! Supplied numerical points project through the original variable's actual
//! definition/reference mapping, never through a second variable identity.
use super::super::*;
use runmat_value::NumericScalar;
pub(in crate::builtins::math::optim::optimmodel) async fn validate_names(
    plan: &super::plan::Plan,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<()> {
    let mut seen = std::collections::HashMap::new();
    for owner in plan.nodes.values() {
        if let Kind::Variable(Value::Object(object)) = &owner.node().kind {
            let view = super::super::super::optimvar::View::new(object, cooperation).await?;
            let name = view.definition().name_owner();
            let target = view.selection().definition();
            if seen.get(name).is_some_and(|previous| *previous != target) {
                return Err(
                    policy.invalid("different original variables have the same public name")
                );
            }
            seen.try_reserve(1)
                .map_err(|source| policy.reserve(source))?;
            seen.insert(name.clone(), target);
        }
        cooperation.completed(1).await?;
    }
    Ok(())
}
pub(super) async fn read(
    reference: &Value,
    point: &std::collections::HashMap<runmat_gc::GcHandle, RootedValue>,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    let Value::Object(reference) = reference else {
        return Err(policy.invalid("variable leaf lost its native reference"));
    };
    let view = super::super::super::optimvar::View::new(reference, cooperation).await?;
    let provided = point.get(&view.selection().definition()).ok_or_else(|| {
        policy.invalid("variable point was not prepared for its original definition")
    })?;
    let provided = provided.value();
    let shape = crate::builtins::common::shape::value_dimensions_borrowed(provided)?;
    if view.selection().is_whole()
        && crate::builtins::common::shape::array_shapes_equal(&shape, view.selection().shape())
    {
        return RootedValue::in_optional_context(provided.clone(), cooperation.runtime());
    }
    let count = view.selection().element_count();
    let mut values =
        runmat_value::exact_capacity_vec(count).map_err(|source| policy.storage(source))?;
    for position in 0..count {
        let index = view
            .selection()
            .definition_index(position)
            .ok_or_else(|| policy.invalid("variable reference exceeds its original definition"))?;
        let value = match provided {
            Value::Num(value) if index == 0 => *value,
            Value::Tensor(value) => {
                let Some(NumericScalar::F64(value)) = value.numeric_value_at(index) else {
                    return Err(policy.invalid("supplied variable has inconsistent double backing"));
                };
                value
            }
            Value::SparseTensor(value) => value
                .get(index % value.rows, index / value.rows)
                .ok_or_else(|| {
                    policy.invalid("supplied variable has inconsistent sparse backing")
                })?,
            _ => return Err(policy.invalid("supplied variable has inconsistent double backing")),
        };
        values.push(value);
        cooperation.completed(1).await?;
    }
    super::state::output(
        values,
        view.selection().shape().clone(),
        policy,
        cooperation.runtime(),
    )
}

/// Validate every selected original field before conversion, then retain one
/// actual double carrier per definition. Subset leaves do not repeatedly cast
/// or copy a whole supplied point.
pub(in crate::builtins::math::optim::optimmodel) async fn prepare(
    plan: &super::plan::Plan,
    point: &Value,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<std::collections::HashMap<runmat_gc::GcHandle, RootedValue>> {
    let Value::Struct(point) = point else {
        return Err(policy
            .invalid("evaluation point must be a scalar structure of original variable values"));
    };
    let mut selected = std::collections::HashMap::new();
    for node in plan.nodes.values() {
        cooperation.completed(1).await?;
        let Kind::Variable(Value::Object(object)) = &node.node().kind else {
            continue;
        };
        let view = super::super::super::optimvar::View::new(object, cooperation).await?;
        let definition = view.selection().definition();
        if selected.contains_key(&definition) {
            continue;
        }
        let provided = point.fields.get(view.definition().name()).ok_or_else(|| {
            policy.invalid("evaluation point is missing a reachable original variable")
        })?;
        let shape = construct::input_shape(provided, policy, cooperation).await?;
        if !crate::builtins::common::shape::array_shapes_equal(&shape, view.definition().shape()) {
            return Err(
                policy.invalid("supplied variable value has the wrong original array shape")
            );
        }
        selected
            .try_reserve(1)
            .map_err(|source| policy.reserve(source))?;
        selected.insert(definition, provided);
    }
    let mut prepared = std::collections::HashMap::new();
    prepared
        .try_reserve(selected.len())
        .map_err(|source| policy.reserve(source))?;
    for (definition, provided) in selected {
        let converted = construct::normalize(provided, policy, cooperation).await?;
        prepared.insert(definition, converted);
        cooperation.completed(1).await?;
    }
    Ok(prepared)
}
