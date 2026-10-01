//! Host-visible admission before provider reads or numerical evaluation.

use super::*;

#[derive(Clone, Copy)]
pub(super) enum EvaluationKind {
    Density,
    Cumulative,
}

pub(super) fn evaluation(
    builtin: &'static str,
    distribution: &Value,
    input: &Value,
    parameters: &[Value],
    kind: EvaluationKind,
) -> BuiltinResult<()> {
    let mut shapes = vec![numeric_shape(builtin, input)?];
    if matches!(distribution, Value::Object(_)) {
        if !parameters.is_empty() {
            return Err(invalid_for(
                builtin,
                format!("{builtin}: fitted-object form accepts one evaluation input"),
            ));
        }
        distribution_from_value(distribution)?;
    } else {
        let family = parse_distribution_name_for(builtin, distribution)?;
        let count = parameters.len();
        let valid = match (kind, family) {
            (
                EvaluationKind::Density | EvaluationKind::Cumulative,
                DistributionKind::Normal | DistributionKind::Lognormal,
            ) => count <= 2,
            _ => count == family.parameter_names().len(),
        };
        if !valid {
            return Err(invalid_for(
                builtin,
                format!(
                    "{builtin}: incorrect parameter count for {}",
                    family.canonical_name()
                ),
            ));
        }
        for parameter in parameters {
            shapes.push(numeric_shape(builtin, parameter)?);
        }
    }
    let mut shape = shapes[0].to_vec();
    for next in shapes.iter().skip(1) {
        shape = broadcast::broadcast_shapes(builtin, &shape, next)
            .map_err(|error| invalid_for(builtin, error))?;
    }
    if crate::builtins::common::spec::checked_reduction_elements(&shape).is_none() {
        return Err(invalid_for(
            builtin,
            format!("{builtin}: output shape exceeds structural range"),
        ));
    }
    Ok(())
}

pub(super) fn fit(data: &Value, distribution: &Value, options: &[Value]) -> BuiltinResult<()> {
    parse_distribution_name(distribution)?;
    let data_shape = numeric_shape(FITDIST_NAME, data)?;
    if !is_vector_shape(data_shape) {
        return Err(invalid("fitdist: observations must form a vector"));
    }
    if !options.len().is_multiple_of(2) {
        return Err(invalid("fitdist: name-value options must be paired"));
    }
    let mut frequency_seen = false;
    for pair in options.chunks_exact(2) {
        let name =
            keyword_of(&pair[0]).ok_or_else(|| invalid("fitdist: option names must be text"))?;
        match FittedFitOption::from_keyword(&name) {
            Some(FittedFitOption::Frequency) if !frequency_seen => {
                frequency_seen = true;
                let weight_shape = numeric_shape(FITDIST_NAME, &pair[1])?;
                if !is_vector_shape(weight_shape)
                    || crate::builtins::common::spec::checked_reduction_elements(data_shape)
                        != crate::builtins::common::spec::checked_reduction_elements(weight_shape)
                {
                    return Err(invalid(
                        "fitdist: Frequency requires one count per observation",
                    ));
                }
            }
            Some(FittedFitOption::Frequency) => {
                return Err(invalid("fitdist: Frequency may be specified only once"))
            }
            Some(_) => {
                return Err(invalid(format!(
                    "fitdist: option '{name}' is not supported yet"
                )))
            }
            None => return Err(invalid(format!("fitdist: unknown option '{name}'"))),
        }
    }
    Ok(())
}

pub(super) fn random(distribution: &Value, rest: &[Value]) -> BuiltinResult<()> {
    if matches!(distribution, Value::Object(_)) {
        distribution_from_value(distribution)?;
        for size in rest {
            if !is_vector_shape(numeric_shape(RANDOM_NAME, size)?) {
                return Err(invalid_for(
                    RANDOM_NAME,
                    "random: size must be scalar or vector",
                ));
            }
        }
        return Ok(());
    }
    let family = parse_distribution_name_for(RANDOM_NAME, distribution)?;
    let parameter_count = family.parameter_names().len();
    if rest.len() < parameter_count {
        return Err(invalid_for(
            RANDOM_NAME,
            format!(
                "random: {} requires {parameter_count} parameters",
                family.canonical_name()
            ),
        ));
    }
    for parameter in &rest[..parameter_count] {
        let shape = numeric_shape(RANDOM_NAME, parameter)?;
        if crate::builtins::common::spec::checked_reduction_elements(shape) != Some(1) {
            return Err(invalid_for(
                RANDOM_NAME,
                "random: named parameters must be scalar",
            ));
        }
    }
    for size in &rest[parameter_count..] {
        if !is_vector_shape(numeric_shape(RANDOM_NAME, size)?) {
            return Err(invalid_for(
                RANDOM_NAME,
                "random: size must be scalar or vector",
            ));
        }
    }
    Ok(())
}

fn is_vector_shape(shape: &[usize]) -> bool {
    shape.iter().filter(|&&extent| extent > 1).count() <= 1
}

fn numeric_shape(builtin: &'static str, value: &Value) -> BuiltinResult<&[usize]> {
    let shape = match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => &[1, 1][..],
        Value::Tensor(tensor) => tensor.shape.as_slice(),
        Value::LogicalArray(array) => array.shape.as_slice(),
        Value::GpuTensor(handle)
            if handle.physical().storage() == runmat_accelerate_api::GpuTensorStorage::Real =>
        {
            handle.shape()
        }
        _ => {
            return Err(invalid_for(
                builtin,
                format!("{builtin}: expected real numeric input"),
            ))
        }
    };
    if crate::builtins::common::spec::checked_reduction_elements(shape).is_none() {
        return Err(invalid_for(
            builtin,
            format!("{builtin}: input shape exceeds structural range"),
        ));
    }
    Ok(shape)
}
