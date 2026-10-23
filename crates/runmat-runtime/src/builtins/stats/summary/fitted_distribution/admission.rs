//! Host-visible admission before provider reads or numerical evaluation.

use super::*;

pub(super) fn evaluation(
    operation: FittedOperation,
    distribution: &Value,
    input: &Value,
    parameters: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<DistributionSelection> {
    let builtin = operation.name();
    let mut shapes = vec![numeric_shape(builtin, input)?];
    let selection = DistributionSelection::from_value(operation, distribution)?;
    match &selection {
        DistributionSelection::Fitted(_) if !parameters.is_empty() => {
            return Err(invalid_for(
                builtin,
                format!("{builtin}: fitted-object form accepts one evaluation input"),
            ));
        }
        DistributionSelection::Fitted(_) => {}
        DistributionSelection::Named(family) => {
            let count = parameters.len();
            let valid = match family {
                DistributionKind::Normal | DistributionKind::Lognormal => count <= 2,
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
    Ok(selection)
}

pub(super) struct FitAdmission<'a> {
    pub(super) kind: DistributionKind,
    pub(super) frequency: Option<&'a Value>,
}

pub(super) fn fit<'a>(
    data: &Value,
    distribution: &Value,
    options: crate::call::FunctionArgumentView<'a>,
) -> BuiltinResult<FitAdmission<'a>> {
    let kind = parse_distribution_name(distribution)?;
    let data_shape = numeric_shape(FITDIST_NAME, data)?;
    if !is_vector_shape(data_shape) {
        return Err(invalid("fitdist: observations must form a vector"));
    }
    if !options.len().is_multiple_of(2) {
        return Err(invalid("fitdist: name-value options must be paired"));
    }
    let mut frequency = None;
    for pair in options.chunks_exact(2) {
        let name =
            keyword_of(&pair[0])?.ok_or_else(|| invalid("fitdist: option names must be text"))?;
        match FittedFitOption::from_keyword(&name) {
            Some(FittedFitOption::Frequency) if frequency.is_none() => {
                let weight_shape = numeric_shape(FITDIST_NAME, &pair[1])?;
                if !is_vector_shape(weight_shape)
                    || crate::builtins::common::spec::checked_reduction_elements(data_shape)
                        != crate::builtins::common::spec::checked_reduction_elements(weight_shape)
                {
                    return Err(invalid(
                        "fitdist: Frequency requires one count per observation",
                    ));
                }
                frequency = pair.get(1);
            }
            Some(FittedFitOption::Frequency) => {
                return Err(invalid("fitdist: Frequency may be specified only once"))
            }
            Some(FittedFitOption::Censoring) => {
                return Err(invalid(
                    "fitdist: Censoring is not supported for fitted distributions yet",
                ))
            }
            Some(FittedFitOption::Options | FittedFitOption::By) => {
                return Err(invalid(format!(
                    "fitdist: option '{name}' is not supported yet"
                )))
            }
            None => return Err(invalid(format!("fitdist: unknown option '{name}'"))),
        }
    }
    Ok(FitAdmission { kind, frequency })
}

pub(super) fn random(
    selection: &DistributionSelection,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    let sizes = match selection {
        DistributionSelection::Fitted(_) => rest,
        DistributionSelection::Named(family) => {
            let parameter_count = selection.parameter_count();
            if rest.len() < parameter_count {
                return Err(invalid_for(
                    RANDOM_NAME,
                    format!(
                        "random: {} requires {parameter_count} parameters",
                        family.canonical_name()
                    ),
                ));
            }
            for parameter in rest.range(..parameter_count)? {
                let shape = numeric_shape(RANDOM_NAME, parameter)?;
                if crate::builtins::common::spec::checked_reduction_elements(shape) != Some(1) {
                    return Err(invalid_for(
                        RANDOM_NAME,
                        "random: named parameters must be scalar",
                    ));
                }
            }
            rest.range(parameter_count..)?
        }
    };
    for size in sizes {
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

fn numeric_shape<'a>(builtin: &'static str, value: &'a Value) -> BuiltinResult<&'a [usize]> {
    let shape = match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => &[1, 1][..],
        Value::Tensor(tensor) => tensor.shape(),
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
