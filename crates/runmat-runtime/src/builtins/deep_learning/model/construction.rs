//! Layer collection, network metadata, and deterministic initialization.
use super::*;

pub(crate) fn is_deep_learning_network_object(object: &ObjectInstance) -> bool {
    object.class_name.is(DLNETWORK_CLASS)
        || object.class_name.is(SERIES_NETWORK_CLASS)
        || object.class_name.is(DAG_NETWORK_CLASS)
}

pub(in crate::builtins) fn layers_from_network_value(
    value: Value,
    function: &'static str,
) -> BuiltinResult<Vec<Value>> {
    match value {
        Value::Object(mut object)
            if object.class_name.is(super::LAYER_GRAPH_CLASS)
                || is_deep_learning_network_object(&object) =>
        {
            validate_sequential_connections(&object, function)?;
            match object.properties.remove("Layers") {
                Some(layers) => layers_from_network_value(layers, function),
                None => Ok(Vec::new()),
            }
        }
        Value::Object(object) => Ok(vec![Value::Object(object)]),
        Value::Cell(cell) => Ok(cell.data),
        Value::OutputList(values) => Ok(values),
        other => Err(deep_learning_error(
            function,
            format!("{function}: layers must be a layer object or cell collection, got {other:?}"),
        )),
    }
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
            if object.class_name.is(super::LAYER_GRAPH_CLASS)
                || is_deep_learning_network_object(object) =>
        {
            validate_sequential_connections(object, function)?;
            match object.properties.get("Layers") {
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
        Value::Cell(cell) => Ok(&cell.data),
        Value::OutputList(values) => Ok(values),
        other => Err(deep_learning_error(
            function,
            format!("{function}: layers must be a layer object or cell collection, got {other:?}"),
        )),
    }
}

pub(in crate::builtins) fn network_object(
    class_name: runmat_types::StaticClassIdentity,
    layers: Vec<Value>,
    mut options: std::collections::BTreeMap<String, Value>,
    function: &'static str,
) -> BuiltinResult<Value> {
    let names = layer_names(&layers, function)?;
    if let Some(requested) = remove_case_insensitive_option(&mut options, "OutputNames") {
        let requested = scalar_text(&requested, function)?;
        if names.last().map(String::as_str) != Some(requested.as_str()) {
            return Err(deep_learning_error(
                function,
                format!("{function}: OutputNames must name the final layer in the supported sequential network"),
            ));
        }
    }
    let layer_count = layers.len();
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
    let initial_state = super::state::initial(&layers, function)?;
    let layers_cell = CellArray::new(layers, layer_count, 1)
        .map(Value::Cell)
        .map_err(|err| deep_learning_error(function, err))?;
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
    if let Some(value) = remove_case_insensitive_option(&mut options, "Initialize") {
        let initialize = logical_scalar(&value, function, "Initialize")?;
        properties.push(("Initialized".to_string(), Value::Bool(initialize)));
        properties.push(("Initialize".to_string(), Value::Bool(initialize)));
    }
    for (name, value) in options {
        properties.push((name, value));
    }
    super::super::class_registration::ensure_network_class_registered(class_name);
    let mut network = super::object_with_identity(class_name, properties);
    let Value::Object(object) = &mut network else {
        unreachable!("network constructor creates an object")
    };
    let requested_initialize = matches!(
        object.properties.get("Initialized"),
        Some(Value::Bool(true))
    );
    let initialized = requested_initialize
        && super::parameters::all_filled(object, function)?
        && super::state::all_filled(object, function)?;
    object
        .properties
        .insert("Initialized".into(), Value::Bool(initialized));
    Ok(network)
}

pub(super) struct NetworkOptions {
    pub(super) initialize: bool,
    explicit_initialize: bool,
    output_names: Option<Value>,
}

impl NetworkOptions {
    pub(super) fn into_properties(self) -> std::collections::BTreeMap<String, Value> {
        let mut properties = std::collections::BTreeMap::new();
        if self.explicit_initialize {
            properties.insert("Initialize".into(), Value::Bool(self.initialize));
        }
        if let Some(output_names) = self.output_names {
            properties.insert("OutputNames".into(), output_names);
        }
        properties
    }
}

pub(super) fn parse_network_options(rest: &[Value]) -> BuiltinResult<NetworkOptions> {
    let mut options = parse_name_values(rest.to_vec(), "dlnetwork")?;
    for name in options.keys() {
        if !matches!(name.as_str(), "initialize" | "outputnames") {
            return Err(deep_learning_error(
                "dlnetwork",
                format!("dlnetwork: unsupported option '{name}'"),
            ));
        }
    }
    let explicit_initialize = options.contains_key("initialize");
    let initialize = options
        .remove("initialize")
        .map(|value| logical_scalar(&value, "dlnetwork", "Initialize"))
        .transpose()?
        .unwrap_or(true);
    let output_names = options.remove("outputnames");
    Ok(NetworkOptions {
        initialize,
        explicit_initialize,
        output_names,
    })
}

fn remove_case_insensitive_option(
    options: &mut std::collections::BTreeMap<String, Value>,
    name: &str,
) -> Option<Value> {
    let key = options
        .keys()
        .find(|key| key.eq_ignore_ascii_case(name))
        .cloned()?;
    options.remove(&key)
}

pub(in crate::builtins) fn initialise_fully_connected_layers(
    layers: &mut [Value],
    function: &'static str,
) -> BuiltinResult<()> {
    let mut current_width = None;
    for layer in layers {
        let Value::Object(object) = layer else {
            continue;
        };
        match ForwardLayerKind::from_identity(&object.class_name) {
            Some(ForwardLayerKind::FeatureInput) => {
                current_width = Some(feature_input_width(object, function)?);
            }
            Some(ForwardLayerKind::FullyConnected) => {
                let input_width = current_width.ok_or_else(|| {
                    deep_learning_error(
                        function,
                        "dlnetwork: fullyConnectedLayer requires a preceding featureInputLayer or fullyConnectedLayer",
                    )
                })?;
                let output_size = positive_property_usize(object, "OutputSize", function)?;
                if !object.properties.contains_key("Weights") {
                    object.properties.insert(
                        "Weights".to_string(),
                        deterministic_weights(output_size, input_width, function)?,
                    );
                }
                if !object.properties.contains_key("Bias") {
                    object.properties.insert(
                        "Bias".to_string(),
                        tensor_value(vec![0.0; output_size], vec![output_size, 1], function)?,
                    );
                }
                current_width = Some(output_size);
            }
            Some(
                ForwardLayerKind::Relu
                | ForwardLayerKind::Elu
                | ForwardLayerKind::Softmax
                | ForwardLayerKind::ClassificationOutput
                | ForwardLayerKind::RegressionOutput,
            ) => {}
            _ => {}
        }
    }
    Ok(())
}

fn deterministic_weights(
    output_size: usize,
    input_width: usize,
    function: &'static str,
) -> BuiltinResult<Value> {
    let count = output_size.checked_mul(input_width).ok_or_else(|| {
        deep_learning_error(
            function,
            format!("{function}: initialized weight shape exceeds addressable storage"),
        )
    })?;
    let mut values = Vec::new();
    values.try_reserve_exact(count).map_err(|_| {
        deep_learning_error(
            function,
            format!("{function}: cannot allocate initialized weights"),
        )
    })?;
    let scale = (input_width as f64).sqrt().max(1.0);
    for col in 0..input_width {
        for row in 0..output_size {
            let signed = ((row + col + 1) % 5) as f64 - 2.0;
            values.push(signed / (10.0 * scale));
        }
    }
    tensor_value(values, vec![output_size, input_width], function)
}
