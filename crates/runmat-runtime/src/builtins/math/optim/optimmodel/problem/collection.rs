//! Shallow public objective/constraint collections. Only their concrete
//! fields are inspected; arbitrary captures/properties are never traversed.
use super::*;
use runmat_gc::GcHandle;

pub(super) async fn normalize(
    value: &Value,
    property: Property,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    if property == Property::Constraints {
        return constraints::collection::normalize(value, None, policy, cooperation).await;
    }
    if property != Property::Objective {
        return Err(policy.invalid("not an objective or constraint collection"));
    }
    if let Value::Struct(fields) = value {
        let mut normalized =
            runmat_value::exact_capacity_vec(fields.fields.len()).map_err(|e| policy.storage(e))?;
        for (name, value) in &fields.fields {
            if !crate::builtins::common::identifiers::is_valid_varname(name) {
                return Err(
                    policy.invalid("objective and constraint names must be valid identifiers")
                );
            }
            // [] removes the named member, not a zero-valued constraint.
            if empty_double(value, policy)? {
                continue;
            }
            let root = objective_leaf(value, policy, cooperation).await?;
            let node = owner::OwnedNode::read(owner::reference(root.value(), policy)?, policy)?;
            if evaluation::element_count(&node.node().shape, policy)? != 1 {
                return Err(policy.invalid("each named objective must be a scalar expression"));
            }
            normalized.push((name, root));
            cooperation.completed(1).await?;
        }
        return crate::execution::structure::construct_struct_from_fields_in_optional_context(
            cooperation.runtime(),
            normalized.iter().map(|(name, root)| (*name, root.value())),
        );
    }
    objective_leaf(value, policy, cooperation).await
}
fn empty_double(value: &Value, policy: Policy) -> BuiltinResult<bool> {
    let Value::Tensor(array) = value else {
        return Ok(false);
    };
    array
        .validate_shape()
        .map_err(|source| policy.invalid(source))?;
    Ok(array.is_empty() && array.numeric_dtype() == runmat_value::NumericDType::F64)
}
fn is_empty_collection(value: &Value, policy: Policy) -> BuiltinResult<bool> {
    Ok(empty_double(value, policy)?
        || matches!(value,Value::StructArray(array) if array.is_empty()&&array.field_names().len()==0))
}
async fn objective_leaf(
    value: &Value,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    if empty_double(value, policy)? {
        construct::empty(
            crate::call::FunctionArgumentView::from_slice(&[]),
            policy,
            cooperation,
        )
        .await
    } else {
        owner::as_expression(value, policy, cooperation).await
    }
}
pub(super) async fn roots(
    model: &Data,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<GcHandle>> {
    let mut roots = Vec::new();
    append(
        &model.objective,
        Property::Objective,
        &mut roots,
        policy,
        cooperation,
    )
    .await?;
    append(
        &model.constraints,
        Property::Constraints,
        &mut roots,
        policy,
        cooperation,
    )
    .await?;
    Ok(roots)
}
async fn append(
    value: &Value,
    property: Property,
    roots: &mut Vec<GcHandle>,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<()> {
    if let Value::Struct(fields) = value {
        for (_, value) in &fields.fields {
            append_leaf(value, property, roots, policy)?;
            cooperation.completed(1).await?;
        }
    } else if property != Property::Constraints || !is_empty_collection(value, policy)? {
        append_leaf(value, property, roots, policy)?;
        cooperation.completed(1).await?;
    }
    Ok(())
}
fn append_leaf(
    value: &Value,
    property: Property,
    roots: &mut Vec<GcHandle>,
    policy: Policy,
) -> BuiltinResult<()> {
    match property {
        Property::Objective => {
            let root = owner::reference(value, policy)?;
            roots.try_reserve(1).map_err(|e| policy.reserve(e))?;
            roots.push(root);
        }
        Property::Constraints => {
            let value = constraints::reference(value, policy)?;
            roots.try_reserve(2).map_err(|e| policy.reserve(e))?;
            roots.extend([value.left, value.right]);
        }
        _ => return Err(policy.invalid("not a model expression collection")),
    }
    Ok(())
}
pub(super) async fn validate_names(
    model: &Data,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<()> {
    let roots = roots(model, policy, cooperation).await?;
    validate_collection_names(
        &roots,
        &[&model.objective, &model.constraints],
        policy,
        cooperation,
    )
    .await
}
pub(in crate::builtins::math::optim::optimmodel) async fn validate_collection_names(
    roots: &[GcHandle],
    collections: &[&Value],
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<()> {
    // This validates conflicting names across all original definition handles,
    // including names present only in separate comparison operands.
    let variables = properties::variables_roots(roots, policy, cooperation).await?;
    let mut names = std::collections::HashSet::new();
    match variables.value() {
        Value::Struct(variables) => {
            names
                .try_reserve(variables.fields.len())
                .map_err(|e| policy.reserve(e))?;
            for (name, _) in &variables.fields {
                names.insert(name);
            }
        }
        Value::StructArray(array) if array.is_empty() && array.field_names().len() == 0 => {}
        _ => return Err(policy.invalid("variable projection lost its structure")),
    }
    for &collection in collections {
        if let Value::Struct(fields) = collection {
            for (name, _) in &fields.fields {
                names.try_reserve(1).map_err(|e| policy.reserve(e))?;
                if !names.insert(name) {
                    return Err(policy
                        .invalid("variable, objective and constraint names must be different"));
                }
                cooperation.completed(1).await?;
            }
        }
    }
    Ok(())
}

pub(super) async fn validate_senses(
    model: &Data,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<()> {
    let Senses::Named(senses) = &model.sense else {
        return Ok(());
    };
    let Value::Struct(objectives) = &model.objective else {
        return Err(policy.invalid("a named ObjectiveSense requires a named Objective structure"));
    };
    if objectives.fields.len() != senses.len() {
        return Err(
            policy.invalid("Objective and ObjectiveSense structures require the same field names")
        );
    }
    let mut seen = std::collections::HashSet::new();
    seen.try_reserve(senses.len())
        .map_err(|e| policy.reserve(e))?;
    for (name, _) in senses {
        if !objectives.fields.contains_key(name.as_str()) || !seen.insert(name.as_str()) {
            return Err(policy
                .invalid("Objective and ObjectiveSense structures require the same field names"));
        }
        cooperation.completed(1).await?;
    }
    Ok(())
}
