//! Real problem-property generations over retained expression/constraint
//! objects. A problem has no second variable table or copied bounds authority.
use super::*;
use runmat_builtins::OptimizationProblemProperty as Property;
use runmat_value::optimization_model::OptimizationObjectiveSense as Sense;

use runmat_value::optimization_model::{
    OptimizationObjectiveSenses as Senses, OptimizationProblemData as Data,
};
pub(super) mod collection;
mod protocol;
mod solve;
pub(super) use solve::execute as solve;
#[cfg(test)]
mod tests;
pub(super) use protocol::{assign, read};

pub(super) fn data(value: &Value, policy: Policy) -> BuiltinResult<&Data> {
    let Value::Object(object) = value else {
        return Err(policy.invalid("expected an optimization problem"));
    };
    if !object.class_name().is(standard::OPTIMIZATION_PROBLEM) || object.selected_class().is_some()
    {
        return Err(policy.invalid("expected an original native optimization problem"));
    }
    let Some(data) = object.optimization_problem_payload() else {
        return Err(policy.invalid("optimization problem has no native model generation"));
    };
    Ok(data)
}
pub(super) fn empty_structure(policy: Policy) -> BuiltinResult<Value> {
    Ok(Value::StructArray(
        runmat_value::StructArray::empty(Vec::new(), vec![0, 0]).map_err(|e| policy.invalid(e))?,
    ))
}
pub(super) async fn create(
    args: crate::call::FunctionArgumentView<'_>,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    let _inputs = policy.roots(args.iter(), cooperation.runtime())?;
    if !args.len().is_multiple_of(2) {
        return Err(policy.invalid("optimproblem expects property name/value pairs"));
    }
    let objective = owner::publish(
        Node {
            shape: SharedShape::new_unmanaged(vec![0, 0]),
            index_names: construct::unnamed(2, policy)?,
            kind: Kind::EmptyInitializer,
        },
        policy,
        cooperation,
    )
    .await?;
    let mut data = Data {
        description: Value::CharArray(runmat_value::CharArray::new_row("")),
        sense: Senses::Uniform(Sense::Minimize),
        objective: objective.value().clone(),
        constraints: empty_structure(policy)?,
    };
    let mut seen = [false; 5];
    // The default objective stays rooted until it has moved into the published
    // problem. Each normalized property likewise gets its own retained root.
    let mut roots =
        runmat_value::exact_capacity_vec(args.len() / 2).map_err(|e| policy.storage(e))?;
    for pair in args.chunks_exact(2) {
        let name = policy.text(&pair[0], cooperation.runtime())?;
        let property = Property::parse_name_value(&name)
            .ok_or_else(|| policy.invalid("unknown problem property"))?;
        let slot = Property::ALL
            .iter()
            .position(|known| *known == property)
            .expect("canonical property");
        if seen[slot] {
            return Err(policy.invalid("duplicate problem property"));
        }
        seen[slot] = true;
        let retained = replace(&mut data, property, &pair[1], policy, cooperation).await?;
        if let Some(root) = retained {
            roots.push(root)
        }
    }
    publish(data, policy, cooperation).await
}
pub(super) async fn publish(
    model: Data,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    registration::ensure(
        standard::OPTIMIZATION_PROBLEM,
        policy,
        cooperation.runtime(),
    )?;
    let source = Value::Object(
        ObjectInstance::new(standard::OPTIMIZATION_PROBLEM.owned())
            .with_optimization_problem_payload(model),
    );
    let _roots = policy.roots(std::iter::once(&source), cooperation.runtime())?;
    let data = data(&source, policy)?;
    collection::validate_names(data, policy, cooperation).await?;
    collection::validate_senses(data, policy, cooperation).await?;
    RootedValue::in_optional_context(source, cooperation.runtime())
}
pub(super) async fn get(
    value: &Value,
    property: Property,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    let current = data(value, policy)?;
    match property {
        Property::Description => {
            RootedValue::in_optional_context(current.description.clone(), cooperation.runtime())
        }
        Property::Objective => {
            RootedValue::in_optional_context(current.objective.clone(), cooperation.runtime())
        }
        Property::Constraints => {
            RootedValue::in_optional_context(current.constraints.clone(), cooperation.runtime())
        }
        Property::Variables => {
            let roots = collection::roots(current, policy, cooperation).await?;
            if roots.is_empty() {
                return RootedValue::in_optional_context(
                    empty_structure(policy)?,
                    cooperation.runtime(),
                );
            }
            let variables = properties::variables_roots(&roots, policy, cooperation).await?;
            if matches!(variables.value(), Value::Struct(fields) if fields.fields.is_empty()) {
                RootedValue::in_optional_context(empty_structure(policy)?, cooperation.runtime())
            } else {
                Ok(variables)
            }
        }
        Property::ObjectiveSense => match &current.sense {
            Senses::Uniform(sense) => RootedValue::in_optional_context(
                Value::String(Utf8Text::new(sense.name().to_owned()).into()),
                cooperation.runtime(),
            ),
            Senses::Named(fields) => {
                let mut roots = runmat_value::exact_capacity_vec(fields.len())
                    .map_err(|e| policy.storage(e))?;
                for (_, sense) in fields {
                    roots.push(RootedValue::in_optional_context(
                        Value::String(Utf8Text::new(sense.name().to_owned()).into()),
                        cooperation.runtime(),
                    )?);
                    cooperation.completed(1).await?;
                }
                crate::execution::structure::construct_struct_from_fields_in_optional_context(
                    cooperation.runtime(),
                    fields
                        .iter()
                        .zip(&roots)
                        .map(|((name, _), value)| (name.as_str(), value.value())),
                )
            }
        },
    }
}
pub(super) async fn set(
    value: &Value,
    property: Property,
    rhs: &Value,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    let _inputs = policy.roots([value, rhs].into_iter(), cooperation.runtime())?;
    let current = data(value, policy)?;
    let sense = match &current.sense {
        Senses::Uniform(sense) => Senses::Uniform(*sense),
        Senses::Named(fields) => {
            let mut copy =
                runmat_value::exact_capacity_vec(fields.len()).map_err(|e| policy.storage(e))?;
            for (name, sense) in fields {
                copy.push((name.clone(), *sense));
                cooperation.completed(1).await?;
            }
            Senses::Named(copy)
        }
    };
    let mut successor = Data {
        description: current.description.clone(),
        sense,
        objective: current.objective.clone(),
        constraints: current.constraints.clone(),
    };
    let _replacement = replace(&mut successor, property, rhs, policy, cooperation).await?;
    publish(successor, policy, cooperation).await
}
async fn replace(
    data: &mut Data,
    property: Property,
    rhs: &Value,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<RootedValue>> {
    match property {
        Property::Variables => return Err(policy.invalid("Variables is a read-only property")),
        Property::Description => {
            // Validation is textual only; preserve the original accepted text
            // owner/class instead of reconstructing it on every property read.
            let _text = policy.text(rhs, cooperation.runtime())?;
            data.description = rhs.clone();
        }
        Property::ObjectiveSense => {
            data.sense = match rhs {
                Value::Struct(fields) => {
                    let mut senses = runmat_value::exact_capacity_vec(fields.fields.len())
                        .map_err(|e| policy.storage(e))?;
                    for (name, value) in &fields.fields {
                        let sense = Sense::parse(&policy.text(value, cooperation.runtime())?)
                            .ok_or_else(|| {
                                policy
                                    .invalid("ObjectiveSense must be minimize/min or maximize/max")
                            })?;
                        senses.push((Utf8Text::new(name.to_owned()), sense));
                        cooperation.completed(1).await?;
                    }
                    Senses::Named(senses)
                }
                _ => Senses::Uniform(
                    Sense::parse(&policy.text(rhs, cooperation.runtime())?).ok_or_else(|| {
                        policy.invalid("ObjectiveSense must be minimize/min or maximize/max")
                    })?,
                ),
            };
        }
        Property::Objective | Property::Constraints => {
            let value = collection::normalize(rhs, property, policy, cooperation).await?;
            if property == Property::Objective {
                data.objective = value.value().clone()
            } else {
                data.constraints = value.value().clone()
            }
            return Ok(Some(value));
        }
    }
    Ok(None)
}
