use super::*;

pub(in crate::builtins) fn layer_object(
    class_name: &str,
    type_name: &str,
    mut properties: Vec<(&str, Value)>,
    rest: Vec<Value>,
    function: &'static str,
) -> BuiltinResult<Value> {
    let mut owned_properties = properties
        .drain(..)
        .map(|(name, value)| (name.to_string(), value))
        .collect::<Vec<_>>();
    let mut name = String::new();
    let mut description = String::new();
    let mut extra = parse_name_values(rest, function)?;
    if let Some(value) = extra.remove("name") {
        name = scalar_text(&value, function)?;
    }
    if let Some(value) = extra.remove("description") {
        description = scalar_text(&value, function)?;
    }
    owned_properties.push(("Type".to_string(), Value::String(type_name.to_string())));
    owned_properties.push(("Name".to_string(), Value::String(name)));
    owned_properties.push(("Description".to_string(), Value::String(description)));
    for (key, value) in extra {
        owned_properties.push((canonical_property_name(&key), value));
    }
    Ok(object(class_name, owned_properties))
}

pub(super) fn canonical_property_name(name: &str) -> String {
    match name.to_ascii_lowercase().as_str() {
        "biaslearnratefactor" => "BiasLearnRateFactor",
        "biasl2factor" => "BiasL2Factor",
        "biasinitializer" => "BiasInitializer",
        "weightslearnratefactor" => "WeightsLearnRateFactor",
        "weightsl2factor" => "WeightsL2Factor",
        "weightsinitializer" => "WeightsInitializer",
        "inputnames" => "InputNames",
        "outputnames" => "OutputNames",
        "padding" => "Padding",
        "stride" => "Stride",
        "dilationfactor" => "DilationFactor",
        "numchannels" => "NumChannels",
        "hasstateinputs" => "HasStateInputs",
        "hasstateoutputs" => "HasStateOutputs",
        "outputmode" => "OutputMode",
        "stateactivationfunction" => "StateActivationFunction",
        "gateactivationfunction" => "GateActivationFunction",
        "normalization" => "Normalization",
        "splitcomplexinputs" => "SplitComplexInputs",
        "weights" => "Weights",
        "bias" => "Bias",
        "classes" => "Classes",
        "epsilon" => "Epsilon",
        "alphalearnratefactor" => "AlphaLearnRateFactor",
        "betalearnratefactor" => "BetaLearnRateFactor",
        "offset" => "Offset",
        "scale" => "Scale",
        other => other,
    }
    .to_string()
}

pub(in crate::builtins) fn parse_name_values(
    args: Vec<Value>,
    function: &'static str,
) -> BuiltinResult<std::collections::BTreeMap<String, Value>> {
    if !args.len().is_multiple_of(2) {
        return Err(deep_learning_error(
            function,
            format!("{function}: name-value options must be paired"),
        ));
    }
    let mut map = std::collections::BTreeMap::new();
    let mut idx = 0;
    while idx < args.len() {
        let name = scalar_text(&args[idx], function)?.to_ascii_lowercase();
        map.insert(name, args[idx + 1].clone());
        idx += 2;
    }
    Ok(map)
}

pub(in crate::builtins) fn layers_from_value(
    value: Value,
    function: &'static str,
) -> BuiltinResult<Vec<Value>> {
    match value {
        Value::Object(_) => Ok(vec![value]),
        Value::Cell(cell) => Ok(cell.data),
        Value::OutputList(values) => Ok(values),
        other => Err(deep_learning_error(
            function,
            format!("{function}: layers must be a layer object, cell array, or object array, got {other:?}"),
        )),
    }
}

pub(in crate::builtins) fn layer_names(
    layers: &[Value],
    function: &'static str,
) -> BuiltinResult<Vec<String>> {
    let mut names = Vec::with_capacity(layers.len());
    for (idx, layer) in layers.iter().enumerate() {
        match layer {
            Value::Object(object) => {
                let name = object
                    .properties
                    .get("Name")
                    .and_then(|value| match value {
                        Value::String(s) if !s.is_empty() => Some(s.clone()),
                        _ => None,
                    })
                    .unwrap_or_else(|| format!("layer_{}", idx + 1));
                names.push(name);
            }
            other => {
                return Err(deep_learning_error(
                    function,
                    format!("{function}: layer list contains non-object value {other:?}"),
                ));
            }
        }
    }
    Ok(names)
}
