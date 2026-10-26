//! Numeric parameters of named built-in operators are controls. Extra
//! arguments of custom callbacks remain opaque and are never gathered here.
use super::{common, Field, Operator, Owner, ScalarCooperation};
use crate::{context::RuntimeContext, BuiltinResult};
use runmat_builtins::*;
use runmat_value::Value;

#[derive(Clone, Copy)]
enum Rule {
    ScalarRatio,
    VectorRatio,
    Gaussian,
    Uniform,
    Tournament,
    Top,
    ShiftLinear,
}
fn rule(owner: Owner, field: Field, value: &Value) -> BuiltinResult<Option<Rule>> {
    if super::super::inputs::raw::unspecified(value) {
        return Ok(None);
    }
    Ok(match field {
        Field::CrossoverFcn => {
            match Operator::parse_for(owner, value, OptimGeneticCrossover::parse)?.kind() {
                Some(OptimGeneticCrossover::Intermediate) => Some(Rule::VectorRatio),
                Some(OptimGeneticCrossover::Heuristic) => Some(Rule::ScalarRatio),
                _ => None,
            }
        }
        Field::MutationFcn => {
            match Operator::parse_for(owner, value, OptimGeneticMutation::parse)?.kind() {
                Some(OptimGeneticMutation::Gaussian) => Some(Rule::Gaussian),
                Some(OptimGeneticMutation::Uniform) => Some(Rule::Uniform),
                _ => None,
            }
        }
        Field::SelectionFcn => {
            match Operator::parse_for(owner, value, OptimGeneticSelection::parse)?.kind() {
                Some(OptimGeneticSelection::Tournament) => Some(Rule::Tournament),
                _ => None,
            }
        }
        Field::FitnessScalingFcn => {
            match Operator::parse_for(owner, value, OptimGeneticScaling::parse)?.kind() {
                Some(OptimGeneticScaling::Top) => Some(Rule::Top),
                Some(OptimGeneticScaling::ShiftLinear) => Some(Rule::ShiftLinear),
                _ => None,
            }
        }
        _ => None,
    })
}
fn extra(value: &Value) -> &[Value] {
    match value {
        Value::Cell(values) => &values.data()[1..],
        _ => &[],
    }
}
pub(super) fn preflight(
    owner: Owner,
    field: Field,
    value: &Value,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<()> {
    let Some(rule) = rule(owner, field, value)? else {
        return Ok(());
    };
    for parameter in extra(value) {
        if !matches!(
            parameter,
            Value::Num(_) | Value::Int(_) | Value::Tensor(_) | Value::GpuTensor(_)
        ) {
            return Err(owner.invalid("built-in operator parameters must be real numeric controls"));
        }
        let fact = crate::value_fact::value_fact(parameter);
        let real = matches!(fact.kind,runmat_types::ValueKindFact::Numeric(n) if n.domain==runmat_types::NumericDomain::Real);
        let shape = if matches!(rule, Rule::VectorRatio) {
            fact.shape.known_dims().is_none_or(|dims| {
                dims.len() <= 2
                    && dims
                        .iter()
                        .filter(|axis| axis.is_some_and(|n| n != 1))
                        .count()
                        <= 1
            })
        } else {
            fact.shape.element_count().is_none_or(|n| n == 1)
        };
        if !real
            || !shape
            || fact.shape.element_count() == Some(0)
            || fact.storage == runmat_types::StorageFact::Sparse
        {
            return Err(owner.invalid("built-in operator parameter has invalid class or geometry"));
        }
        common::preflight_host_residency(owner.name(), parameter, None, runtime)?;
        if common::is_typed_integer_value(parameter) {
            let extension = match owner {
                Owner::Constructor => &OPTIMOPTIONS_INTEGER_OPTION_EXTENSION,
                Owner::Solver => &GA_INTEGER_OPTIONS_EXTENSION,
            };
            crate::compatibility::ensure_builtin_extension_enabled_in_optional_context(
                runtime,
                extension,
                owner.name(),
            )?;
        }
    }
    Ok(())
}
pub(super) async fn materialize(
    field: Field,
    value: &Value,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<Value>> {
    if rule(Owner::Solver, field, value)?.is_none() || extra(value).is_empty() {
        return Ok(None);
    }
    let Value::Cell(source) = value else {
        unreachable!("parameterized operator cell");
    };
    let mut values = super::super::POLICY.capacity(source.data().len(), cooperation.runtime())?;
    values.push(source.data()[0].clone());
    for parameter in extra(value) {
        values.push(
            crate::dispatcher::gather_if_needed_in_context_async(parameter, cooperation.runtime())
                .await?,
        );
        cooperation.completed(1).await?;
    }
    Ok(Some(Value::Cell(
        runmat_value::CellArray::new_with_shape(values, source.shape().to_vec())
            .map_err(|error| Owner::Solver.invalid(error))?,
    )))
}
pub(super) async fn validate(
    owner: Owner,
    field: Field,
    value: &Value,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<()> {
    let Some(rule) = rule(owner, field, value)? else {
        return Ok(());
    };
    for (index, parameter) in extra(value).iter().enumerate() {
        // A constructor can retain admitted resident controls for the selected
        // solver; it must not change the transfer/effect policy of the field.
        if matches!(parameter, Value::GpuTensor(_)) {
            continue;
        }
        if matches!(rule, Rule::Tournament) {
            if common::option_usize_value(owner.name(), Some(parameter), "tournament size", 0)? < 2
            {
                return Err(owner.invalid("tournament size must be at least two"));
            }
        } else if let Value::Tensor(tensor) = parameter {
            for element in 0..tensor.len() {
                let value = if let Some(storage) = tensor.integer_storage() {
                    storage
                        .value_at(element)
                        .and_then(|n| n.to_exact_f64())
                        .ok_or_else(|| {
                            owner.invalid(
                                "operator parameter integer is not exactly representable as double",
                            )
                        })?
                } else {
                    crate::builtins::common::tensor::tensor_value_f64(tensor, element)
                };
                check(owner, rule, index, value)?;
                cooperation.completed(1).await?;
            }
        } else {
            check(owner, rule, index, super::real(owner, parameter)?)?;
        }
        cooperation.completed(1).await?;
    }
    Ok(())
}
fn check(owner: Owner, rule: Rule, index: usize, value: f64) -> BuiltinResult<()> {
    let valid = value.is_finite()
        && match rule {
            Rule::ScalarRatio | Rule::VectorRatio => value >= 0.,
            Rule::Gaussian => index != 0 || value >= 0.,
            Rule::Uniform => (0.0..=1.0).contains(&value),
            Rule::Top => value >= 0. && (value <= 1. || value.fract() == 0.),
            Rule::ShiftLinear => value >= 1.,
            Rule::Tournament => unreachable!("exact count owner"),
        };
    if valid {
        Ok(())
    } else {
        Err(owner.invalid("built-in operator parameter is outside its numerical domain"))
    }
}

pub(super) async fn crossover_ratio(
    operator: &Operator<OptimGeneticCrossover>,
    variables: usize,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<f64>> {
    let parameter = if matches!(
        operator.kind(),
        Some(OptimGeneticCrossover::Intermediate | OptimGeneticCrossover::Heuristic)
    ) {
        operator.extra().first()
    } else {
        None
    };
    let count = parameter
        .map(|value| match value {
            Value::Tensor(tensor) => tensor.len(),
            _ => 1,
        })
        .unwrap_or(0);
    if count != 0 && count != 1 && count != variables {
        return Err(
            Owner::Solver.invalid("crossover ratio must be scalar or contain nvars entries")
        );
    }
    let mut result = super::super::POLICY.capacity(count, cooperation.runtime())?;
    if let Some(Value::Tensor(tensor)) = parameter {
        for index in 0..tensor.len() {
            let value = if let Some(storage) = tensor.integer_storage() {
                storage
                    .value_at(index)
                    .and_then(|value| value.to_exact_f64())
                    .ok_or_else(|| {
                        Owner::Solver
                            .invalid("crossover ratio integer cannot be represented as double")
                    })?
            } else {
                crate::builtins::common::tensor::tensor_value_f64(tensor, index)
            };
            result.push(value);
            cooperation.completed(1).await?;
        }
    } else if let Some(value) = parameter {
        result.push(super::real(Owner::Solver, value)?);
        cooperation.completed(1).await?;
    }
    Ok(result)
}
