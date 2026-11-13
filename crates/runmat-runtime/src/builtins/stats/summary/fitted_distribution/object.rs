use super::*;

pub(super) fn distribution_object(fit: &FittedDistribution) -> BuiltinResult<ObjectInstance> {
    let mut object = runmat_value::UnboundedObjectBuilder::new(
        runmat_types::standard::PROBABILITY_DISTRIBUTION.to_string(),
    );
    object.insert(
        "DistributionName".to_string(),
        Value::string(fit.kind.canonical_name().to_string()),
    );
    object.insert(
        "DistName".to_string(),
        Value::string(fit.kind.canonical_name().to_string()),
    );
    object.insert(
        "ParameterNames".to_string(),
        Value::StringArray(string_row(
            fit.kind
                .parameter_names()
                .iter()
                .map(|name| (*name).to_string())
                .collect(),
        )?),
    );
    object.insert(
        "ParameterValues".to_string(),
        Value::Tensor(
            Tensor::new(fit.parameters.clone(), vec![1, fit.parameters.len()])
                .map_err(|err| internal(format!("fitdist: {err}")))?,
        ),
    );
    object.insert(
        "NumParameters".to_string(),
        Value::Num(fit.parameters.len() as f64),
    );
    object.insert("NumObservations".to_string(), Value::Num(fit.observations));
    object.insert("NLogL".to_string(), Value::Num(fit.nlogl));
    object.insert("IsTruncated".to_string(), Value::Bool(false));
    for (name, value) in fit.kind.parameter_names().iter().zip(fit.parameters.iter()) {
        object.insert((*name).to_string(), Value::Num(*value));
    }
    Ok(object.finish())
}

pub(super) fn string_row(values: Vec<String>) -> BuiltinResult<StringArray> {
    StringArray::new(values.clone(), vec![1, values.len()])
        .map_err(|err| internal(format!("fitdist: {err}")))
}

pub(super) fn distribution_from_value_for(
    operation: FittedOperation,
    value: &Value,
) -> BuiltinResult<FittedDistribution> {
    let builtin = operation.name();
    let Value::Object(object) = value else {
        return Err(invalid_for(
            builtin,
            format!("{builtin}: expected ProbabilityDistribution object"),
        ));
    };
    if !object.is_class(runmat_types::standard::PROBABILITY_DISTRIBUTION) {
        return Err(invalid_for(
            builtin,
            format!(
                "{builtin}: expected ProbabilityDistribution object, got {}",
                object.class_name()
            ),
        ));
    }
    let dist_name = string_property(builtin, object, "DistributionName")?;
    let kind = parse_distribution_keyword_for(builtin, &dist_name)?;
    let parameters = numeric_vector_property(builtin, object, "ParameterValues")?;
    if parameters.len() != kind.parameter_names().len() {
        return Err(invalid_for(
            builtin,
            format!("{builtin}: ProbabilityDistribution object has malformed ParameterValues"),
        ));
    }
    let nlogl = numeric_scalar_property(object, "NLogL").unwrap_or(f64::NAN);
    let observations = numeric_scalar_property(object, "NumObservations").unwrap_or(f64::NAN);
    Ok(FittedDistribution {
        kind,
        parameters,
        nlogl,
        observations,
    })
}

pub(super) fn string_property(
    builtin: &'static str,
    object: &ObjectInstance,
    name: &str,
) -> BuiltinResult<String> {
    match object.properties().get(name) {
        Some(Value::String(runmat_value::StringElement::Text(value))) => {
            Ok(value.as_str().to_owned())
        }
        Some(Value::CharArray(chars)) if chars.rows == 1 => {
            crate::builtins::stats::decode_utf16_with_error(chars.data(), || {
                invalid_for(
                    builtin,
                    format!("{builtin}: ProbabilityDistribution {name} is not valid UTF-16 text"),
                )
            })
        }
        _ => Err(invalid_for(
            builtin,
            format!("{builtin}: ProbabilityDistribution object is missing {name}"),
        )),
    }
}

pub(super) fn numeric_vector_property(
    builtin: &'static str,
    object: &ObjectInstance,
    name: &str,
) -> BuiltinResult<Vec<f64>> {
    match object.properties().get(name) {
        Some(Value::Tensor(tensor)) => Ok(tensor::tensor_values_f64(tensor)),
        Some(Value::Num(value)) => Ok(vec![*value]),
        _ => Err(invalid_for(
            builtin,
            format!("{builtin}: ProbabilityDistribution object is missing {name}"),
        )),
    }
}

pub(super) fn numeric_scalar_property(object: &ObjectInstance, name: &str) -> Option<f64> {
    match object.properties().get(name) {
        Some(Value::Num(value)) => Some(*value),
        _ => None,
    }
}

pub(super) fn parse_distribution_name(value: &Value) -> BuiltinResult<DistributionKind> {
    parse_distribution_name_for(FITDIST_NAME, value)
}

pub(super) fn parse_distribution_name_for(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<DistributionKind> {
    let keyword = keyword_of(value)?.ok_or_else(|| {
        invalid_for(
            builtin,
            format!("{builtin}: distribution name must be a string scalar"),
        )
    })?;
    parse_distribution_keyword_for(builtin, &keyword)
}

pub(super) fn parse_distribution_keyword_for(
    builtin: &'static str,
    keyword: &str,
) -> BuiltinResult<DistributionKind> {
    DistributionKind::from_keyword(keyword).ok_or_else(|| {
        invalid_for(
            builtin,
            format!("{builtin}: unsupported distribution '{keyword}'"),
        )
    })
}
