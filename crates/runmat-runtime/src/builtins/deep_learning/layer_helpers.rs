use super::*;
use runmat_builtins::{DeepLearningLayerKind, DeepLearningLayerOption};
use std::collections::BTreeMap;

pub(in crate::builtins::deep_learning) mod options;

pub(in crate::builtins) fn layer_object(
    class_name: runmat_types::StaticClassIdentity,
    type_name: &str,
    mut properties: Vec<(&str, Value)>,
    mut extra: BTreeMap<DeepLearningLayerOption, Value>,
    function: &'static str,
) -> BuiltinResult<Value> {
    let mut owned_properties = properties
        .drain(..)
        .map(|(name, value)| (name.to_string(), value))
        .collect::<Vec<_>>();
    let mut name = String::new();
    let mut description = String::new();
    if DeepLearningLayerKind::from_identity(&class_name.owned()).is_none() {
        return Err(deep_learning_error(
            function,
            format!("{function}: unknown layer class"),
        ));
    }
    if let Some(value) = extra.remove(&DeepLearningLayerOption::Name) {
        name = scalar_text(&value, function)?;
    }
    if let Some(value) = extra.remove(&DeepLearningLayerOption::Description) {
        description = scalar_text(&value, function)?;
    }
    owned_properties.push(("Type".to_string(), Value::string(type_name.to_string())));
    owned_properties.push(("Name".to_string(), Value::string(name)));
    owned_properties.push(("Description".to_string(), Value::string(description)));
    for (key, value) in extra {
        owned_properties.push((key.canonical_name().to_string(), value));
    }
    object_with_identity(class_name, owned_properties)
}

pub(super) fn parse_layer_options(
    args: crate::call::FunctionArgumentView<'_>,
    kind: DeepLearningLayerKind,
    function: &'static str,
) -> BuiltinResult<BTreeMap<DeepLearningLayerOption, Value>> {
    if !args.len().is_multiple_of(2) {
        return Err(deep_learning_error(
            function,
            format!("{function}: name-value options must be paired"),
        ));
    }
    let mut parsed = BTreeMap::new();
    for pair in args.chunks_exact(2) {
        let name = scalar_text(&pair[0], function)?;
        let option = kind.option(&name).ok_or_else(|| {
            deep_learning_error(
                function,
                format!("{function}: {name} is not a supported option for this layer"),
            )
        })?;
        options::validate(kind, option, &pair[1], function)?;
        if parsed.insert(option, pair[1].clone()).is_some() {
            return Err(deep_learning_error(
                function,
                format!(
                    "{function}: {} is specified more than once",
                    option.canonical_name()
                ),
            ));
        }
    }
    Ok(parsed)
}

pub(in crate::builtins) fn parse_name_values(
    args: crate::call::BuiltinArguments,
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

pub(in crate::builtins) fn layers_from_value<'a>(
    value: &'a Value,
    function: &'static str,
) -> BuiltinResult<&'a [Value]> {
    super::model::layers_from_network_ref(value, function)
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
                    .properties()
                    .get("Name")
                    .and_then(|value| match value {
                        Value::String(runmat_value::StringElement::Text(s)) if !s.is_empty() => {
                            Some(s.to_string())
                        }
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
