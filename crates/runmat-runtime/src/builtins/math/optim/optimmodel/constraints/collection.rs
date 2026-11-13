//! Native comparison collections share normalization and original edge discovery.
use super::*;
use runmat_gc::GcHandle;
fn empty_double(value: &Value, policy: Policy) -> BuiltinResult<bool> {
    if let Value::Tensor(array) = value {
        array.validate_shape().map_err(|e| policy.invalid(e))?;
        return Ok(array.is_empty() && array.numeric_dtype() == runmat_value::NumericDType::F64);
    }
    Ok(false)
}
pub(in crate::builtins::math::optim::optimmodel) fn empty(
    value: &Value,
    policy: Policy,
) -> BuiltinResult<bool> {
    Ok(empty_double(value, policy)?
        || matches!(value,Value::StructArray(array) if array.is_empty() && array.field_names().len()==0))
}
fn leaf(
    value: &Value,
    relation: Option<Relation>,
    policy: Policy,
) -> BuiltinResult<&OptimizationConstraintRef> {
    let reference = super::reference(value, policy)?;
    if relation.is_some_and(|required| required != reference.relation) {
        return Err(policy.invalid("equation problems require original equality values"));
    }
    Ok(reference)
}
pub(in crate::builtins::math::optim::optimmodel) async fn normalize(
    value: &Value,
    relation: Option<Relation>,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    if let Value::Struct(fields) = value {
        let mut roots =
            runmat_value::exact_capacity_vec(fields.fields.len()).map_err(|e| policy.storage(e))?;
        for (name, value) in &fields.fields {
            if !crate::builtins::common::identifiers::is_valid_varname(name) {
                return Err(policy.invalid("comparison names must be valid identifiers"));
            }
            if empty_double(value, policy)? {
                continue;
            }
            let _ = leaf(value, relation, policy)?;
            roots.push((
                name,
                RootedValue::in_optional_context(value.clone(), c.runtime())?,
            ));
            c.completed(1).await?;
        }
        return crate::execution::structure::construct_struct_from_fields_in_optional_context(
            c.runtime(),
            roots.iter().map(|(name, value)| (*name, value.value())),
        );
    }
    if empty(value, policy)? {
        return RootedValue::in_optional_context(problem::empty_structure(policy)?, c.runtime());
    }
    let _ = leaf(value, relation, policy)?;
    RootedValue::in_optional_context(value.clone(), c.runtime())
}
pub(in crate::builtins::math::optim::optimmodel) async fn roots(
    value: &Value,
    relation: Option<Relation>,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Vec<GcHandle>> {
    let mut roots = Vec::new();
    if let Value::Struct(fields) = value {
        roots
            .try_reserve(
                fields
                    .fields
                    .len()
                    .checked_mul(2)
                    .ok_or_else(|| policy.invalid("comparison root count overflow"))?,
            )
            .map_err(|e| policy.reserve(e))?;
        for (_, value) in &fields.fields {
            let leaf = leaf(value, relation, policy)?;
            roots.extend([leaf.left, leaf.right]);
            c.completed(1).await?;
        }
    } else if !empty(value, policy)? {
        let leaf = leaf(value, relation, policy)?;
        roots.try_reserve(2).map_err(|e| policy.reserve(e))?;
        roots.extend([leaf.left, leaf.right]);
        c.completed(1).await?;
    }
    Ok(roots)
}
