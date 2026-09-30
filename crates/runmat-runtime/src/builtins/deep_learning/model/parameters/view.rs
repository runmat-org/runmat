//! Borrowed parameter enumeration and cross-network topology admission.
use super::*;
use std::collections::HashSet;

fn layers<'a>(network: &'a ObjectInstance, function: &'static str) -> BuiltinResult<&'a [Value]> {
    let stored = network.properties.get("Layers").ok_or_else(|| {
        deep_learning_error(function, format!("{function}: network is missing Layers"))
    })?;
    let Value::Cell(cell) = stored else {
        return Err(deep_learning_error(
            function,
            format!("{function}: Layers must be a cell array"),
        ));
    };
    if cell.shape != [cell.data.len(), 1] {
        return Err(deep_learning_error(
            function,
            format!("{function}: Layers must be a column cell array"),
        ));
    }
    Ok(&cell.data)
}

/// Borrow the parameters consumed by supported sequential execution without
/// materializing host payloads or downloading GPU buffers.
pub(super) fn view<'a>(
    network: &'a ObjectInstance,
    function: &'static str,
) -> BuiltinResult<Vec<ParameterView<'a>>> {
    if !is_deep_learning_network_object(network) {
        return Err(deep_learning_error(
            function,
            format!("{function}: expected a network object"),
        ));
    }
    let layers = layers(network, function)?;
    validate_forward_layers(layers, function)?;
    let names = layer_names(layers, function)?;
    if names.iter().collect::<HashSet<_>>().len() != names.len() {
        return Err(deep_learning_error(
            function,
            format!("{function}: layer names must be unique"),
        ));
    }
    let mut width = None;
    let mut out = Vec::new();
    for (index, layer) in layers.iter().enumerate() {
        let Value::Object(layer) = layer else {
            unreachable!("validated layers")
        };
        match ForwardLayerKind::from_identity(&layer.class_name) {
            Some(ForwardLayerKind::FeatureInput) => {
                // A vector InputSize can be stored by deferred construction;
                // it does not give one flattened execution width here.
                let input_size = layer.properties.get("InputSize").ok_or_else(|| {
                    deep_learning_error(
                        function,
                        format!("{function}: featureInputLayer is missing InputSize"),
                    )
                })?;
                let dimensions = numeric_vector(input_size, function, "InputSize")?;
                width = if dimensions.len() == 1 {
                    Some(dimensions[0])
                } else {
                    None
                };
            }
            Some(ForwardLayerKind::FullyConnected) => {
                let output_width = positive_property_usize(&layer, "OutputSize", function)?;
                for (role, expected_shape) in [
                    (
                        ParameterRole::Weights,
                        ParameterShape {
                            rows: output_width,
                            columns: width,
                        },
                    ),
                    (
                        ParameterRole::Bias,
                        ParameterShape {
                            rows: output_width,
                            columns: Some(1),
                        },
                    ),
                ] {
                    let value = layer.properties.get(role.name());
                    if let Some(value) = value {
                        validate_parameter(value, &expected_shape, role, function)?;
                    }
                    out.push(ParameterView {
                        key: ParameterKey {
                            layer_index: index,
                            layer_name: names[index].clone(),
                            role,
                        },
                        value,
                        expected_shape,
                    });
                }
                width = Some(output_width);
            }
            _ => {}
        }
    }
    Ok(out)
}

pub(in crate::builtins) fn validate_matching_networks(
    networks: &[&ObjectInstance],
    function: &'static str,
) -> BuiltinResult<()> {
    let Some(first) = networks.first() else {
        return Ok(());
    };
    let expected = view(first, function)?;
    let first_layers = layers(first, function)?;
    for network in networks.iter().skip(1) {
        if network.class_name != first.class_name {
            return Err(deep_learning_error(
                function,
                format!("{function}: network classes differ"),
            ));
        }
        let actual = view(network, function)?;
        let actual_layers = layers(network, function)?;
        if first_layers.len() != actual_layers.len()
            || first_layers.iter().zip(actual_layers).any(|(left, right)| {
                let (Value::Object(left), Value::Object(right)) = (left, right) else {
                    return true;
                };
                left.class_name != right.class_name
                    || left.properties.get("Name") != right.properties.get("Name")
                    || left.properties.get("InputSize") != right.properties.get("InputSize")
                    || left.properties.get("OutputSize") != right.properties.get("OutputSize")
            })
        {
            return Err(deep_learning_error(
                function,
                format!("{function}: network layer topology differs"),
            ));
        }
        if actual.len() != expected.len()
            || actual
                .iter()
                .zip(&expected)
                .any(|(a, b)| a.key != b.key || a.expected_shape != b.expected_shape)
        {
            return Err(deep_learning_error(
                function,
                format!("{function}: network parameter topology differs"),
            ));
        }
    }
    Ok(())
}
