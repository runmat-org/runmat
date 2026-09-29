use super::*;

#[derive(Debug)]
pub(super) struct AdmittedExport {
    pub(super) network: ObjectInstance,
    pub(super) filename: String,
    pub(super) options: ExportOptions,
    pub(super) layers: Vec<Value>,
}

/// Validate every host-decidable input before acquiring any provider-owned data.
/// Only the option values and learnables consumed by the exporter are downloaded.
pub(super) async fn admit_export_call(args: Vec<Value>) -> BuiltinResult<AdmittedExport> {
    if args.len() < 2 {
        return Err(deep_learning_error(
            EXPORT_ONNX,
            "exportONNXNetwork: expected network and filename",
        ));
    }
    let network = require_network_object(&args[0])?.clone();
    let filename = scalar_text(&args[1], EXPORT_ONNX)?;
    if filename.is_empty() {
        return Err(deep_learning_error(
            EXPORT_ONNX,
            "exportONNXNetwork: filename must not be empty",
        ));
    }
    let mut options_map = parse_name_values(args[2..].to_vec(), EXPORT_ONNX)?;
    if options_map.iter().any(|(key, value)| {
        matches!(key.as_str(), "opsetversion" | "batchsize" | "verbose")
            && crate::builtins::common::validation::value_has_native_integer_class(value)
    }) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &runmat_builtins::EXPORT_ONNX_INTEGER_EXTENSION,
            EXPORT_ONNX,
        )?;
    }
    for (key, value) in &options_map {
        validate_option(key, value)?;
    }
    let layers = model::layers_from_network_value(Value::Object(network.clone()), EXPORT_ONNX)?;
    preflight_layers(&network, &layers, &options_map)?;

    for value in options_map.values_mut() {
        if matches!(value, Value::GpuTensor(_)) {
            *value = crate::gather_if_needed_async(value).await?;
        }
    }
    let options = parse_export_options(options_map)?;
    let mut prepared = Vec::with_capacity(layers.len());
    for layer in layers {
        let Value::Object(mut object) = layer else {
            unreachable!("preflight_layers verified object identity")
        };
        if matches!(
            ForwardLayerKind::from_identity(&object.class_name),
            Some(ForwardLayerKind::FullyConnected)
        ) {
            for property in ["Weights", "Bias"] {
                let value = object
                    .properties
                    .get(property)
                    .expect("preflight verified learnable property");
                let gathered = gather_learnable(value).await?;
                object.properties.insert(property.to_string(), gathered);
            }
        }
        prepared.push(Value::Object(object));
    }
    Ok(AdmittedExport {
        network,
        filename,
        options,
        layers: prepared,
    })
}

fn preflight_layers(
    network: &ObjectInstance,
    layers: &[Value],
    options: &std::collections::BTreeMap<String, Value>,
) -> BuiltinResult<()> {
    if layers.is_empty() {
        return Err(deep_learning_error(
            EXPORT_ONNX,
            "exportONNXNetwork: network must contain at least one layer",
        ));
    }
    let names = super::layer_names(layers, EXPORT_ONNX)?;
    let mut seen = BTreeSet::new();
    for name in names {
        if !seen.insert(name.clone()) {
            return Err(deep_learning_error(
                EXPORT_ONNX,
                format!("exportONNXNetwork: duplicate layer name '{name}'"),
            ));
        }
    }
    if !options.contains_key("inputnames") {
        graph::network_name(network, "InputNames")?;
    }
    if !options.contains_key("outputnames") {
        graph::network_name(network, "OutputNames")?;
    }
    let mut current_features = None;
    for (index, layer) in layers.iter().enumerate() {
        let Value::Object(layer) = layer else {
            return Err(deep_learning_error(
                EXPORT_ONNX,
                "exportONNXNetwork: network layers must be layer objects",
            ));
        };
        match ForwardLayerKind::from_identity(&layer.class_name) {
            Some(ForwardLayerKind::FeatureInput) if index == 0 => {
                current_features = Some(model::feature_input_width(layer, EXPORT_ONNX)?);
            }
            Some(ForwardLayerKind::FullyConnected) if index > 0 => {
                let input_features = current_features.ok_or_else(|| deep_learning_error(
                    EXPORT_ONNX,
                    "exportONNXNetwork: fullyConnectedLayer requires a preceding featureInputLayer",
                ))?;
                let output_size = model::positive_property_usize(layer, "OutputSize", EXPORT_ONNX)?;
                let mut shapes = Vec::with_capacity(2);
                for property in ["Weights", "Bias"] {
                    let value = layer.properties.get(property).ok_or_else(|| {
                        deep_learning_error(
                            EXPORT_ONNX,
                            format!("exportONNXNetwork: layer is missing {property}"),
                        )
                    })?;
                    shapes.push(preflight_learnable(value, property)?);
                }
                let weights = &shapes[0];
                let weight_rows = weights.first().copied().unwrap_or(0);
                let weight_cols = weights.get(1).copied().unwrap_or(1);
                if weights.len() > 2 || weight_rows != output_size || weight_cols != input_features {
                    return Err(deep_learning_error(EXPORT_ONNX, format!("exportONNXNetwork: fullyConnectedLayer Weights must be {output_size}-by-{input_features}")));
                }
                let bias_len = shapes[1].iter().try_fold(1usize, |count, extent| count.checked_mul(*extent))
                    .ok_or_else(|| deep_learning_error(EXPORT_ONNX, "exportONNXNetwork: Bias element count exceeds addressable storage"))?;
                if bias_len != output_size {
                    return Err(deep_learning_error(EXPORT_ONNX, format!("exportONNXNetwork: fullyConnectedLayer Bias must have {output_size} elements")));
                }
                current_features = Some(output_size);
            }
            Some(ForwardLayerKind::Elu) if index > 0 => {
                if let Some(value) = layer.properties.get("Alpha") {
                    super::super::numeric_scalar(value, EXPORT_ONNX, "Alpha")?;
                }
            }
            Some(ForwardLayerKind::Relu | ForwardLayerKind::Softmax) if index > 0 => {}
            Some(kind) if kind.is_output() && index + 1 == layers.len() => {}
            Some(ForwardLayerKind::FeatureInput) => {
                return Err(deep_learning_error(
                    EXPORT_ONNX,
                    "exportONNXNetwork: featureInputLayer must be the first exported layer",
                ))
            }
            Some(kind) if kind.is_output() => {
                return Err(deep_learning_error(
                    EXPORT_ONNX,
                    "exportONNXNetwork: output layers must be terminal layers",
                ))
            }
            Some(_) => {
                return Err(deep_learning_error(
                    EXPORT_ONNX,
                    "exportONNXNetwork: network must start with a featureInputLayer",
                ))
            }
            None => {
                return Err(unsupported_error(
                    EXPORT_ONNX,
                    format!(
                        "exportONNXNetwork: layer type '{}' is not supported for RunMat forward execution",
                        layer.class_name
                    ),
                ))
            }
        }
    }
    if !matches!(layers.first(), Some(Value::Object(first)) if first.class_name.is(super::super::FEATURE_INPUT_LAYER_CLASS))
    {
        return Err(deep_learning_error(
            EXPORT_ONNX,
            "exportONNXNetwork: network must start with a featureInputLayer",
        ));
    }
    Ok(())
}

fn learnable_data(value: &Value) -> BuiltinResult<&Value> {
    match value {
        Value::Object(object) if object.class_name.is(super::DLARRAY_CLASS) => {
            object.properties.get("Data").ok_or_else(|| {
                deep_learning_error(EXPORT_ONNX, "exportONNXNetwork: dlarray is missing Data")
            })
        }
        other => Ok(other),
    }
}

fn preflight_learnable(value: &Value, property: &str) -> BuiltinResult<Vec<usize>> {
    match learnable_data(value)? {
        Value::GpuTensor(handle) => {
            if runmat_accelerate_api::handle_is_logical(handle)
                || runmat_accelerate_api::handle_storage(handle)
                    != runmat_accelerate_api::GpuTensorStorage::Real
            {
                return Err(deep_learning_error(
                    EXPORT_ONNX,
                    format!("exportONNXNetwork: {property} must contain real numeric values"),
                ));
            }
            Ok(handle.shape().to_vec())
        }
        _ => {
            let value = super::super::autodiff::dlarray_data(value, EXPORT_ONNX)?;
            let tensor = crate::builtins::common::tensor::value_into_tensor_for(EXPORT_ONNX, value)
                .map_err(|err| {
                    deep_learning_error(EXPORT_ONNX, format!("exportONNXNetwork: {property} {err}"))
                })?;
            Ok(tensor.shape)
        }
    }
}

async fn gather_learnable(value: &Value) -> BuiltinResult<Value> {
    match value {
        Value::Object(object) if object.class_name.is(super::DLARRAY_CLASS) => {
            let data = learnable_data(value)?;
            let gathered = crate::gather_if_needed_async(data).await?;
            let mut object = object.clone();
            object.properties.insert("Data".to_string(), gathered);
            Ok(Value::Object(object))
        }
        _ => Ok(crate::gather_if_needed_async(value).await?),
    }
}

pub(super) fn require_network_object(value: &Value) -> BuiltinResult<&ObjectInstance> {
    match value {
        Value::Object(object) if model::is_deep_learning_network_object(object) => Ok(object),
        other => Err(deep_learning_error(
            EXPORT_ONNX,
            format!(
                "exportONNXNetwork: expected a dlnetwork or trained network object, got {other:?}"
            ),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_accelerate_api::HostTensorView;
    use runmat_value::{CellArray, IntegerStorage};

    fn network(weights: Value, output_size: Value) -> Value {
        let input = super::super::super::object(
            "featureInputLayer",
            [
                ("InputSize", Value::Num(2.0)),
                ("Name", Value::String("input".into())),
            ],
        );
        let dense = super::super::super::object(
            "fullyConnectedLayer",
            [
                ("OutputSize", output_size),
                ("Weights", weights),
                (
                    "Bias",
                    Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap()),
                ),
                ("Name", Value::String("dense".into())),
            ],
        );
        super::super::super::object(
            "dlnetwork",
            [(
                "Layers",
                Value::Cell(CellArray::new(vec![input, dense], 2, 1).unwrap()),
            )],
        )
    }

    #[test]
    fn malformed_host_layer_control_precedes_nested_learnable_download() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let weights = provider
                .upload(&HostTensorView {
                    data: &[1.0, 2.0],
                    shape: &[1, 2],
                })
                .unwrap();
            let net = network(Value::GpuTensor(weights), Value::Num(0.0));
            provider.reset_telemetry();
            let error = futures::executor::block_on(admit_export_call(vec![
                net,
                Value::String("unused.onnx".into()),
            ]))
            .unwrap_err();
            assert!(error.message.contains("OutputSize"));
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        });
    }

    #[test]
    fn nested_learnable_is_downloaded_once_and_unrelated_metadata_is_not() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let weights = provider
                .upload(&HostTensorView {
                    data: &[1.0, 2.0],
                    shape: &[1, 2],
                })
                .unwrap();
            let unrelated = provider
                .upload(&HostTensorView {
                    data: &[7.0],
                    shape: &[1, 1],
                })
                .unwrap();
            let mut net = network(Value::GpuTensor(weights), Value::Num(1.0));
            if let Value::Object(object) = &mut net {
                object
                    .properties
                    .insert("State".into(), Value::GpuTensor(unrelated));
            }
            provider.reset_telemetry();
            let admitted = futures::executor::block_on(admit_export_call(vec![
                net,
                Value::String("unused.onnx".into()),
            ]))
            .unwrap();
            let graph =
                graph::build_export_graph(&admitted.network, &admitted.layers, &admitted.options)
                    .unwrap();
            assert_eq!(graph.initializers.len(), 2);
            assert_eq!(provider.telemetry_snapshot().download_bytes, 16);
        });
    }

    #[test]
    fn typed_integer_control_obeys_runmat_extension_gate() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let weights = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap());
        let opset =
            Value::Tensor(Tensor::new_integer(IntegerStorage::U16(vec![14]), vec![1, 1]).unwrap());
        let error = futures::executor::block_on(admit_export_call(vec![
            network(weights, Value::Num(1.0)),
            Value::String("unused.onnx".into()),
            Value::String("OpsetVersion".into()),
            opset,
        ]))
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:ExportOnnxIntegerControlExtension")
        );
    }
}
