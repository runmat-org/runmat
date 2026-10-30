//! Borrowed parameter enumeration and cross-network topology admission.
use super::*;
#[cfg(test)]
use crate::builtins::deep_learning::arguments::DeepLearningObjectEdit as _;
use runmat_builtins::DeepLearningLayerKind;
use std::collections::HashSet;

#[derive(Clone, Copy)]
enum InputGeometry {
    Feature,
    VectorSequence,
    SpatialSequence,
}

fn push_parameter<'a>(
    out: &mut Vec<ParameterView<'a>>,
    layer: &'a ObjectInstance,
    layer_index: usize,
    layer_name: &str,
    role: ParameterRole,
    expected_shape: ParameterShape,
    function: &'static str,
) -> BuiltinResult<()> {
    let value = layer.properties().get(role.name());
    if let Some(value) = value {
        validate_parameter(value, &expected_shape, role, function)?;
    }
    out.push(ParameterView {
        key: ParameterKey {
            layer_index,
            layer_name: layer_name.to_string(),
            role,
        },
        value,
        expected_shape,
    });
    Ok(())
}

fn width_option(
    layer: &ObjectInstance,
    property: &'static str,
    channels: Option<usize>,
    function: &'static str,
) -> BuiltinResult<Option<usize>> {
    match layer.properties().get(property) {
        None => Ok(channels),
        Some(Value::String(runmat_value::StringElement::Text(value)))
            if value.eq_ignore_ascii_case("auto") =>
        {
            Ok(channels)
        }
        Some(value) => {
            let declared = positive_usize(value, function, property)?;
            if channels.is_some_and(|known| known != declared) {
                return Err(deep_learning_error(
                    function,
                    format!("{function}: {property} conflicts with the preceding layer"),
                ));
            }
            Ok(Some(declared))
        }
    }
}

fn layer_metadata_matches(left: &ObjectInstance, right: &ObjectInstance) -> bool {
    let metadata = |(name, _): &(&str, &Value)| ParameterRole::parse(name).is_none();
    left.class_name() == right.class_name()
        && left.properties().iter().filter(metadata).count()
            == right.properties().iter().filter(metadata).count()
        && left
            .properties()
            .iter()
            .filter(metadata)
            .all(|(name, value)| right.properties().get(name) == Some(value))
}

fn layers<'a>(network: &'a ObjectInstance, function: &'static str) -> BuiltinResult<&'a [Value]> {
    let stored = network.properties().get("Layers").ok_or_else(|| {
        deep_learning_error(function, format!("{function}: network is missing Layers"))
    })?;
    let Value::Cell(cell) = stored else {
        return Err(deep_learning_error(
            function,
            format!("{function}: Layers must be a cell array"),
        ));
    };
    if cell.shape != [cell.data().len(), 1] {
        return Err(deep_learning_error(
            function,
            format!("{function}: Layers must be a column cell array"),
        ));
    }
    Ok(cell.data())
}

/// Borrow the parameters consumed by supported sequential execution without
/// materializing host payloads or downloading GPU buffers.
pub(in crate::builtins::deep_learning) fn view<'a>(
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
    view_layers(layers, function)
}

/// Inspect layer payloads before a network object exists. Construction and
/// execution share the same parameter geometry and storage admission.
pub(in crate::builtins::deep_learning) fn view_layers<'a>(
    layers: &'a [Value],
    function: &'static str,
) -> BuiltinResult<Vec<ParameterView<'a>>> {
    validate_forward_layers(layers, function)?;
    let names = layer_names(layers, function)?;
    if names.iter().collect::<HashSet<_>>().len() != names.len() {
        return Err(deep_learning_error(
            function,
            format!("{function}: layer names must be unique"),
        ));
    }
    let mut channels = None;
    let mut input_geometry = None;
    let mut out = Vec::new();
    for (index, layer) in layers.iter().enumerate() {
        let Value::Object(layer) = layer else {
            unreachable!("validated layers")
        };
        match DeepLearningLayerKind::from_identity(layer.class_name()) {
            Some(DeepLearningLayerKind::FeatureInput | DeepLearningLayerKind::SequenceInput) => {
                // A vector InputSize can be stored by deferred construction;
                // it does not give one flattened execution width here.
                let input_size = layer.properties().get("InputSize").ok_or_else(|| {
                    deep_learning_error(
                        function,
                        format!("{function}: featureInputLayer is missing InputSize"),
                    )
                })?;
                let dimensions = numeric_vector(input_size, function, "InputSize")?;
                input_geometry = match DeepLearningLayerKind::from_identity(layer.class_name()) {
                    Some(DeepLearningLayerKind::FeatureInput) => Some(InputGeometry::Feature),
                    _ if dimensions.len() == 1 => Some(InputGeometry::VectorSequence),
                    _ => Some(InputGeometry::SpatialSequence),
                };
                channels = match input_geometry {
                    Some(InputGeometry::Feature | InputGeometry::VectorSequence) => {
                        (dimensions.len() == 1).then_some(dimensions[0])
                    }
                    Some(InputGeometry::SpatialSequence) => dimensions.last().copied(),
                    None => None,
                };
            }
            Some(DeepLearningLayerKind::FullyConnected) => {
                let output_width = positive_property_usize(layer, "OutputSize", function)?;
                for (role, expected_shape) in [
                    (
                        ParameterRole::Weights,
                        ParameterShape::Matrix {
                            rows: output_width,
                            columns: channels,
                        },
                    ),
                    (ParameterRole::Bias, ParameterShape::Elements(output_width)),
                ] {
                    push_parameter(
                        &mut out,
                        layer,
                        index,
                        &names[index],
                        role,
                        expected_shape,
                        function,
                    )?;
                }
                channels = Some(output_width);
            }
            Some(DeepLearningLayerKind::Convolution1D) => {
                let filter = positive_property_usize(layer, "FilterSize", function)?;
                let filters = positive_property_usize(layer, "NumFilters", function)?;
                channels = width_option(layer, "NumChannels", channels, function)?;
                push_parameter(
                    &mut out,
                    layer,
                    index,
                    &names[index],
                    ParameterRole::Weights,
                    ParameterShape::Dimensions(vec![Some(filter), channels, Some(filters)]),
                    function,
                )?;
                push_parameter(
                    &mut out,
                    layer,
                    index,
                    &names[index],
                    ParameterRole::Bias,
                    ParameterShape::Dimensions(vec![Some(1), Some(filters)]),
                    function,
                )?;
                channels = Some(filters);
            }
            Some(DeepLearningLayerKind::Lstm | DeepLearningLayerKind::Bilstm) => {
                channels = width_option(layer, "InputSize", channels, function)?;
                let units = positive_property_usize(layer, "NumHiddenUnits", function)?;
                let bidirectional = matches!(
                    DeepLearningLayerKind::from_identity(layer.class_name()),
                    Some(DeepLearningLayerKind::Bilstm)
                );
                let gates = units
                    .checked_mul(if bidirectional { 8 } else { 4 })
                    .ok_or_else(|| {
                        deep_learning_error(
                            function,
                            format!("{function}: recurrent parameter shape overflows"),
                        )
                    })?;
                for (role, shape) in [
                    (
                        ParameterRole::InputWeights,
                        ParameterShape::Matrix {
                            rows: gates,
                            columns: channels,
                        },
                    ),
                    (
                        ParameterRole::RecurrentWeights,
                        ParameterShape::Matrix {
                            rows: gates,
                            columns: Some(units),
                        },
                    ),
                    (
                        ParameterRole::Bias,
                        ParameterShape::Dimensions(vec![Some(gates), Some(1)]),
                    ),
                ] {
                    push_parameter(&mut out, layer, index, &names[index], role, shape, function)?;
                }
                channels = units.checked_mul(if bidirectional { 2 } else { 1 });
            }
            Some(DeepLearningLayerKind::LayerNormalization) => {
                channels = width_option(layer, "NumChannels", channels, function)?;
                let shape = match input_geometry {
                    Some(InputGeometry::SpatialSequence) => vec![Some(1), channels],
                    _ => vec![channels, Some(1)],
                };
                for role in [ParameterRole::Scale, ParameterRole::Offset] {
                    push_parameter(
                        &mut out,
                        layer,
                        index,
                        &names[index],
                        role,
                        ParameterShape::Dimensions(shape.clone()),
                        function,
                    )?;
                }
            }
            Some(DeepLearningLayerKind::GlobalAveragePooling1D) => {
                input_geometry = match input_geometry {
                    Some(InputGeometry::SpatialSequence) => Some(InputGeometry::VectorSequence),
                    _ => Some(InputGeometry::Feature),
                };
            }
            Some(
                DeepLearningLayerKind::Relu
                | DeepLearningLayerKind::Elu
                | DeepLearningLayerKind::Softmax
                | DeepLearningLayerKind::ClassificationOutput
                | DeepLearningLayerKind::RegressionOutput,
            ) => {}
            None => {
                return Err(deep_learning_error(
                    function,
                    format!(
                        "{function}: unsupported layer class '{}'",
                        layer.class_name()
                    ),
                ));
            }
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
        if network.class_name() != first.class_name() {
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
                !layer_metadata_matches(left, right)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_comparison_uses_keys_and_values_not_hash_map_iteration_order() {
        let mut left = ObjectInstance::new(DeepLearningLayerKind::FullyConnected.class().owned());
        left.replace_dl_property("Name", Value::String("dense".into()))
            .expect("fixture property");
        left.replace_dl_property("Description", Value::String("projection".into()))
            .expect("fixture property");
        left.replace_dl_property("OutputSize", Value::Num(2.0))
            .expect("fixture property");
        left.replace_dl_property("InputSize", Value::Num(3.0))
            .expect("fixture property");
        left.replace_dl_property("Weights", Value::Num(1.0))
            .expect("fixture property");
        for _ in 0..32 {
            let mut right = ObjectInstance::new(left.class_name().clone());
            for (name, value) in left.properties().iter() {
                right
                    .replace_dl_property(name, value.clone())
                    .expect("fixture property");
            }
            right
                .replace_dl_property("Weights", Value::Num(9.0))
                .expect("fixture property");
            assert!(layer_metadata_matches(&left, &right));
            right
                .replace_dl_property("OutputSize", Value::Num(4.0))
                .expect("fixture property");
            assert!(!layer_metadata_matches(&left, &right));
            right
                .remove_dl_property("OutputSize")
                .expect("fixture removal");
            assert!(!layer_metadata_matches(&left, &right));
        }
    }
}
