//! Layer collection and network metadata.
use super::*;
use crate::builtins::deep_learning::arguments::DeepLearningObjectEdit as _;
use runmat_builtins::DeepLearningNetworkOption;

pub(crate) fn is_deep_learning_network_object(object: &ObjectInstance) -> bool {
    object.class_name().is(DLNETWORK_CLASS)
        || object.class_name().is(SERIES_NETWORK_CLASS)
        || object.class_name().is(DAG_NETWORK_CLASS)
}

pub(in crate::builtins) fn layers_from_network_value(
    value: Value,
    function: &'static str,
) -> BuiltinResult<(CellArray, RootedValueLease)> {
    let layers = layers_from_network_ref(&value, function)?;
    let context = crate::context::legacy::active();
    let result = crate::execution::structure::cell::from_values(
        context.as_ref(),
        &[layers.len(), 1],
        layers,
    )?;
    let result = result.into_value_and_lease();
    let roots = result.1;
    let Value::Cell(layers) = result.0 else {
        unreachable!("Cell constructor")
    };
    Ok((layers, roots))
}

/// Borrow only the consumed layer list. In particular, a network's other
/// properties and its resident learnables are not cloned merely to inspect
/// layer metadata before placement is selected.
pub(in crate::builtins) fn layers_from_network_ref<'a>(
    value: &'a Value,
    function: &'static str,
) -> BuiltinResult<&'a [Value]> {
    match value {
        Value::Object(object)
            if object.class_name().is(super::LAYER_GRAPH_CLASS)
                || is_deep_learning_network_object(object) =>
        {
            validate_sequential_connections(object, function)?;
            match object.properties().get("Layers") {
                Some(layers) => layer_slice(layers, function),
                None => Ok(&[]),
            }
        }
        other => layer_slice(other, function),
    }
}

pub(super) fn layer_slice<'a>(
    value: &'a Value,
    function: &'static str,
) -> BuiltinResult<&'a [Value]> {
    match value {
        Value::Object(_) => Ok(std::slice::from_ref(value)),
        Value::Cell(cell) => Ok(cell.data()),
        Value::OutputList(values) => Ok(values.as_slice()),
        other => Err(deep_learning_error(
            function,
            format!("{function}: layers must be a layer object or cell collection, got {other:?}"),
        )),
    }
}

#[cfg(test)]
pub(in crate::builtins) fn network_object(
    class_name: runmat_types::StaticClassIdentity,
    layers: Vec<Value>,
    options: std::collections::BTreeMap<String, Value>,
    function: &'static str,
) -> BuiltinResult<Value> {
    let context = crate::context::legacy::active();
    let layers = crate::execution::structure::cell::from_values(
        context.as_ref(),
        &[layers.len(), 1],
        &layers,
    )?;
    layers.with_owned(|layers| {
        let Value::Cell(layers) = layers else {
            unreachable!("Cell constructor")
        };
        network_object_from_layer_cell(class_name, layers, options, function)
    })
}

pub(in crate::builtins) fn network_object_from_layer_cell(
    class_name: runmat_types::StaticClassIdentity,
    layers: CellArray,
    mut options: std::collections::BTreeMap<String, Value>,
    function: &'static str,
) -> BuiltinResult<Value> {
    let context = crate::context::legacy::active();
    let _layer_roots = crate::execution::structure::replacement::lease(
        &Value::Cell(layers.clone()),
        context.as_ref(),
    )?;
    let layers_data = layers.data();
    let names = layer_names(layers_data, function)?;
    if let Some(requested) =
        remove_case_insensitive_option(&mut options, DeepLearningNetworkOption::OutputNames)
    {
        validate_output_name(&names, &requested, function)?;
    }
    let layer_count = layers_data.len();
    let input_names = names
        .first()
        .cloned()
        .map(|name| vec![name])
        .unwrap_or_default();
    let output_names = names
        .last()
        .cloned()
        .map(|name| vec![name])
        .unwrap_or_default();
    let requested_initialize =
        remove_case_insensitive_option(&mut options, DeepLearningNetworkOption::Initialize)
            .map(|value| logical_scalar(&value, function, "Initialize"))
            .transpose()?;
    let initial_state = super::state::initial_with_defaults(
        layers_data,
        requested_initialize.unwrap_or(true),
        function,
    )?;
    let layers_cell = Value::Cell(layers);
    let mut properties = vec![
        ("Layers".to_string(), layers_cell),
        (
            "LayerNames".to_string(),
            string_array(names.clone(), vec![layer_count, 1], function)?,
        ),
        (
            "InputNames".to_string(),
            string_array(
                input_names,
                vec![1, names.first().map_or(0, |_| 1)],
                function,
            )?,
        ),
        (
            "OutputNames".to_string(),
            string_array(
                output_names,
                vec![1, names.last().map_or(0, |_| 1)],
                function,
            )?,
        ),
        ("Connections".to_string(), Value::Struct(StructValue::new())),
        (super::state::STORAGE_PROPERTY.to_string(), initial_state),
        ("Initialized".to_string(), Value::Bool(true)),
        (
            "SupportedExecution".to_string(),
            Value::String("sequential-feedforward".into()),
        ),
    ];
    if let Some(initialize) = requested_initialize {
        properties.push(("Initialized".to_string(), Value::Bool(initialize)));
        properties.push(("Initialize".to_string(), Value::Bool(initialize)));
    }
    for (name, value) in options {
        properties.push((name, value));
    }
    super::super::class_registration::ensure_network_class_registered(class_name);
    let mut network = super::object_with_identity(class_name, properties)?;
    let Value::Object(object) = &mut network else {
        unreachable!("network constructor creates an object")
    };
    let requested_initialize = matches!(
        object.properties().get("Initialized"),
        Some(Value::Bool(true))
    );
    let initialized = requested_initialize
        && super::parameters::all_filled(object, function)?
        && super::state::all_filled(object, function)?;
    object.replace_dl_property("Initialized", Value::Bool(initialized))?;
    Ok(network)
}

pub(super) struct NetworkOptions {
    pub(super) initialize: bool,
    explicit_initialize: bool,
    output_names: Option<Value>,
}

impl NetworkOptions {
    pub(super) fn preflight_layers(
        &self,
        layers: &[Value],
        function: &'static str,
    ) -> BuiltinResult<()> {
        if let Some(requested) = &self.output_names {
            validate_output_name(&layer_names(layers, function)?, requested, function)?;
        }
        Ok(())
    }

    pub(super) fn into_properties(self) -> std::collections::BTreeMap<String, Value> {
        let mut properties = std::collections::BTreeMap::new();
        if self.explicit_initialize {
            properties.insert(
                DeepLearningNetworkOption::Initialize.name().into(),
                Value::Bool(self.initialize),
            );
        }
        if let Some(output_names) = self.output_names {
            properties.insert(
                DeepLearningNetworkOption::OutputNames.name().into(),
                output_names,
            );
        }
        properties
    }
}

fn validate_output_name(
    names: &[String],
    requested: &Value,
    function: &'static str,
) -> BuiltinResult<()> {
    let requested = scalar_text(requested, function)?;
    if names.last().map(String::as_str) != Some(requested.as_str()) {
        return Err(deep_learning_error(
            function,
            format!("{function}: OutputNames must name the final layer in the supported sequential network"),
        ));
    }
    Ok(())
}

pub(super) fn parse_network_options(
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<NetworkOptions> {
    if !rest.len().is_multiple_of(2) {
        return Err(deep_learning_error(
            "dlnetwork",
            "dlnetwork: name-value options must be paired",
        ));
    }
    let mut options = std::collections::BTreeMap::new();
    let mut unknown_options = std::collections::BTreeSet::new();
    for pair in rest.chunks_exact(2) {
        let name = scalar_text(&pair[0], "dlnetwork")?;
        if let Some(option) = DeepLearningNetworkOption::from_name(&name) {
            // Repeated name-value keys retain the existing last-value-wins policy.
            options.insert(option, pair[1].clone());
        } else {
            unknown_options.insert(name.to_ascii_lowercase());
        }
    }
    if let Some(name) = unknown_options.into_iter().next() {
        return Err(deep_learning_error(
            "dlnetwork",
            format!("dlnetwork: unsupported option '{name}'"),
        ));
    }
    let explicit_initialize = options.contains_key(&DeepLearningNetworkOption::Initialize);
    let initialize = options
        .remove(&DeepLearningNetworkOption::Initialize)
        .map(|value| {
            logical_scalar(
                &value,
                "dlnetwork",
                DeepLearningNetworkOption::Initialize.name(),
            )
        })
        .transpose()?
        .unwrap_or(true);
    let output_names = options.remove(&DeepLearningNetworkOption::OutputNames);
    Ok(NetworkOptions {
        initialize,
        explicit_initialize,
        output_names,
    })
}

fn remove_case_insensitive_option(
    options: &mut std::collections::BTreeMap<String, Value>,
    option: DeepLearningNetworkOption,
) -> Option<Value> {
    let key = options
        .keys()
        .find(|key| DeepLearningNetworkOption::from_name(key) == Some(option))
        .cloned()?;
    options.remove(&key)
}
