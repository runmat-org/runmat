//! A real equation problem generation, not an optimization objective surrogate.
use super::*;
use runmat_builtins::{EquationProblemProperty as Property, EQUATION_PROBLEM_CLASS};
use runmat_value::optimization_model::EquationProblemData as Data;
use runmat_value::optimization_model::OptimizationConstraintRelation as Relation;
mod protocol;
mod solve;
pub(super) use protocol::{assign, read};
pub(super) use solve::execute as solve;
pub(super) fn data(value: &Value, policy: Policy) -> BuiltinResult<&Data> {
    let Value::Object(object) = value else {
        return Err(policy.invalid("expected an equation problem"));
    };
    if !object.class_name().is(EQUATION_PROBLEM_CLASS) || object.selected_class().is_some() {
        return Err(policy.invalid("expected an original native equation problem"));
    }
    object
        .equation_problem_payload()
        .ok_or_else(|| policy.invalid("equation problem has no native generation"))
}
fn empty(policy: Policy) -> BuiltinResult<Value> {
    Ok(Value::Tensor(
        runmat_value::Tensor::new(Vec::new(), vec![0, 0]).map_err(|e| policy.invalid(e))?,
    ))
}
pub(super) async fn create(
    args: crate::call::FunctionArgumentView<'_>,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    let _inputs = policy.roots(args.iter(), c.runtime())?;
    if !args.len().is_multiple_of(2) {
        return Err(policy.invalid("eqnproblem expects property name/value pairs"));
    }
    let mut value = Data {
        description: Value::CharArray(runmat_value::CharArray::new_row("")),
        equations: empty(policy)?,
    };
    let mut seen = [false; 3];
    let mut roots =
        runmat_value::exact_capacity_vec(args.len() / 2).map_err(|e| policy.storage(e))?;
    for pair in args.chunks_exact(2) {
        let name = policy.text(&pair[0], c.runtime())?;
        let property = Property::parse_name_value(&name)
            .ok_or_else(|| policy.invalid("unknown equation problem property"))?;
        let slot = Property::ALL
            .iter()
            .position(|&p| p == property)
            .expect("canonical equation property");
        if seen[slot] {
            return Err(policy.invalid("duplicate equation problem property"));
        }
        seen[slot] = true;
        if let Some(root) = replace(&mut value, property, &pair[1], policy, c).await? {
            roots.push(root);
        }
        c.completed(1).await?;
    }
    publish(value, policy, c).await
}
async fn publish(
    value: Data,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    super::super::common::native_class_registration::ensure(
        EQUATION_PROBLEM_CLASS,
        &[],
        policy,
        c.runtime(),
    )?;
    let value = Value::Object(
        ObjectInstance::new(EQUATION_PROBLEM_CLASS.owned()).with_equation_problem_payload(value),
    );
    let _inputs = policy.roots(std::iter::once(&value), c.runtime())?;
    let current = data(&value, policy)?;
    let roots =
        constraints::collection::roots(&current.equations, Some(Relation::Equal), policy, c)
            .await?;
    problem::collection::validate_collection_names(&roots, &[], policy, c).await?;
    RootedValue::in_optional_context(value, c.runtime())
}
pub(super) async fn get(
    value: &Value,
    property: Property,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    let current = data(value, policy)?;
    match property {
        Property::Description => {
            RootedValue::in_optional_context(current.description.clone(), c.runtime())
        }
        Property::Equations => {
            RootedValue::in_optional_context(current.equations.clone(), c.runtime())
        }
        Property::Variables => {
            let roots = constraints::collection::roots(
                &current.equations,
                Some(Relation::Equal),
                policy,
                c,
            )
            .await?;
            properties::variables_roots(&roots, policy, c).await
        }
    }
}
pub(super) async fn set(
    value: &Value,
    property: Property,
    rhs: &Value,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    let _inputs = policy.roots([value, rhs].into_iter(), c.runtime())?;
    let current = data(value, policy)?;
    let mut next = Data {
        description: current.description.clone(),
        equations: current.equations.clone(),
    };
    let _replacement = replace(&mut next, property, rhs, policy, c).await?;
    publish(next, policy, c).await
}
async fn replace(
    value: &mut Data,
    property: Property,
    rhs: &Value,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Option<RootedValue>> {
    match property {
        Property::Variables => return Err(policy.invalid("Variables is a read-only property")),
        Property::Description => {
            let _ = policy.text(rhs, c.runtime())?;
            value.description = rhs.clone();
        }
        Property::Equations => {
            let root = if let Value::Tensor(array) = rhs {
                array.validate_shape().map_err(|e| policy.invalid(e))?;
                if array.is_empty() && array.numeric_dtype() == runmat_value::NumericDType::F64 {
                    RootedValue::in_optional_context(rhs.clone(), c.runtime())?
                } else {
                    constraints::collection::normalize(rhs, Some(Relation::Equal), policy, c)
                        .await?
                }
            } else {
                constraints::collection::normalize(rhs, Some(Relation::Equal), policy, c).await?
            };
            value.equations = root.value().clone();
            return Ok(Some(root));
        }
    }
    Ok(None)
}
