use super::*;
use crate::builtins::deep_learning::arguments::DeepLearningObjectEdit as _;

#[derive(Debug)]
pub(super) struct AdmittedExport {
    pub(super) network: ObjectInstance,
    pub(super) filename: String,
    pub(super) options: ExportOptions,
    pub(super) layers: Vec<Value>,
}

/// Validate every host-decidable input before acquiring any provider-owned data.
/// Only the option values and learnables consumed by the exporter are downloaded.
pub(super) async fn admit_export_call(
    mut args: crate::call::BuiltinArguments,
) -> BuiltinResult<AdmittedExport> {
    if args.len() < 2 {
        return Err(deep_learning_error(
            EXPORT_ONNX,
            "exportONNXNetwork: expected network and filename",
        ));
    }
    let option_args = args.range(2..)?;
    let network = require_network_object(&args[0])?;
    let filename = scalar_text(&args[1], EXPORT_ONNX)?;
    if filename.is_empty() {
        return Err(deep_learning_error(
            EXPORT_ONNX,
            "exportONNXNetwork: filename must not be empty",
        ));
    }
    let parsed_options = parse_option_inputs(option_args)?;
    if parsed_options.iter().any(|option| {
        option
            .key
            .is_some_and(ExportOnnxOptionKey::accepts_integer_extension)
            && crate::builtins::common::validation::value_has_native_integer_class(&option.value)
    }) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &runmat_builtins::EXPORT_ONNX_INTEGER_EXTENSION,
            EXPORT_ONNX,
        )?;
    }
    let mut options_map = std::collections::BTreeMap::new();
    for option in parsed_options {
        let key = option.key.ok_or_else(|| unsupported_option(&option.name))?;
        validate_option(key, &option.value)?;
        options_map.insert(key, option.value);
    }
    let layers = model::layers_from_network_ref(&args[0], EXPORT_ONNX)?;
    preflight_layers(network, layers, &options_map)?;

    for value in options_map.values_mut() {
        if matches!(value, Value::GpuTensor(_)) {
            *value = crate::gather_if_needed_async(value).await?;
        }
    }
    let options = parse_export_options(options_map)?;
    // The model metadata stays with the owned object; move its consumed Layers
    // collection into preparation instead of copying every learned parameter.
    let Value::Object(mut network) = args.pop_front().expect("validated argument prefix") else {
        unreachable!("network identity was admitted above")
    };
    let layers_value = network
        .remove_dl_property("Layers")?
        .expect("nonempty Layers collection was admitted above");
    let prepared_layers = model::layers_from_network_value(layers_value, EXPORT_ONNX)?;
    let _layer_roots = prepared_layers.1;
    let layers = prepared_layers.0;
    let mut prepared = Vec::with_capacity(layers.data().len());
    for layer in layers.data().iter().cloned() {
        let Value::Object(mut object) = layer else {
            unreachable!("preflight_layers verified object identity")
        };
        if matches!(
            ForwardLayerKind::from_identity(object.class_name()),
            Some(ForwardLayerKind::FullyConnected)
        ) {
            for property in ["Weights", "Bias"] {
                let value = object
                    .property(property)
                    .cloned()
                    .expect("preflight verified learnable property");
                let gathered = gather_learnable(value).await?;
                object.replace_dl_property(property, gathered)?;
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
    options: &std::collections::BTreeMap<ExportOnnxOptionKey, Value>,
) -> BuiltinResult<()> {
    model::validate_sequential_connections(network, EXPORT_ONNX)?;
    if layers.is_empty() {
        return Err(deep_learning_error(
            EXPORT_ONNX,
            "exportONNXNetwork: network must contain at least one layer",
        ));
    }
    model::validate_sequential_topology(layers, EXPORT_ONNX)?;
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
    if !options.contains_key(&ExportOnnxOptionKey::InputNames) {
        graph::network_name(network, "InputNames")?;
    }
    if !options.contains_key(&ExportOnnxOptionKey::OutputNames) {
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
        match ForwardLayerKind::from_identity(layer.class_name()) {
            Some(ForwardLayerKind::FeatureInput) if index == 0 => {
                current_features = Some(model::feature_input_width(layer, EXPORT_ONNX)?);
            }
            Some(ForwardLayerKind::FullyConnected) if index > 0 => {
                let input_features = current_features.ok_or_else(|| deep_learning_error(
                    EXPORT_ONNX,
                    "exportONNXNetwork: fullyConnectedLayer requires a preceding featureInputLayer",
                ))?;
                current_features = Some(model::fully_connected_output_width(
                    layer,
                    input_features,
                    EXPORT_ONNX,
                )?);
            }
            Some(ForwardLayerKind::Elu) if index > 0 => {
                if let Some(value) = layer.properties().get("Alpha") {
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
                return Err(unsupported_error(
                    EXPORT_ONNX,
                    format!(
                        "exportONNXNetwork: layer type '{}' has no supported ONNX lowering",
                        layer.class_name()
                    ),
                ))
            }
            None => {
                return Err(unsupported_error(
                    EXPORT_ONNX,
                    format!(
                        "exportONNXNetwork: layer type '{}' is not supported for RunMat forward execution",
                        layer.class_name()
                    ),
                ))
            }
        }
    }
    if !matches!(layers.first(), Some(Value::Object(first)) if first.class_name().is(super::super::FEATURE_INPUT_LAYER_CLASS))
    {
        return Err(deep_learning_error(
            EXPORT_ONNX,
            "exportONNXNetwork: network must start with a featureInputLayer",
        ));
    }
    Ok(())
}

async fn gather_learnable(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Object(mut object) if object.class_name().is(super::DLARRAY_CLASS) => {
            let data = object.property("Data").cloned().ok_or_else(|| {
                deep_learning_error(EXPORT_ONNX, "exportONNXNetwork: dlarray is missing Data")
            })?;
            let gathered = gather_owned_data(data).await?;
            object.replace_dl_property("Data", gathered)?;
            Ok(Value::Object(object))
        }
        other => gather_owned_data(other).await,
    }
}

async fn gather_owned_data(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => crate::gather_if_needed_async(&Value::GpuTensor(handle)).await,
        other => Ok(other),
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
            runmat_builtins::FEATURE_INPUT_LAYER_CLASS.display_name(),
            [
                ("InputSize", Value::Num(2.0)),
                ("Name", Value::String("input".into())),
            ],
        )
        .expect("fixture object");
        let dense = super::super::super::object(
            runmat_builtins::FULLY_CONNECTED_LAYER_CLASS.display_name(),
            [
                ("OutputSize", output_size),
                ("Weights", weights),
                (
                    "Bias",
                    Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap()),
                ),
                ("Name", Value::String("dense".into())),
            ],
        )
        .expect("fixture object");
        super::super::super::object(
            model::DLNETWORK_CLASS.display_name(),
            [(
                "Layers",
                Value::Cell(CellArray::new(vec![input, dense], 2, 1).unwrap()),
            )],
        )
        .expect("fixture object")
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
            let error = futures::executor::block_on(admit_export_call(
                crate::builtins::common::test_support::builtin_arguments(vec![
                    net,
                    Value::String("unused.onnx".into()),
                ]),
            ))
            .unwrap_err();
            assert!(error.message.contains("OutputSize"));
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        });
    }

    #[test]
    fn resident_nonscalar_options_reject_before_any_export_download() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            for option in ["OpsetVersion", "BatchSize", "Verbose"] {
                let control = provider
                    .upload(&HostTensorView {
                        data: &[1.0, 2.0],
                        shape: &[1, 2],
                    })
                    .unwrap();
                let weights = provider
                    .upload(&HostTensorView {
                        data: &[1.0, 2.0],
                        shape: &[1, 2],
                    })
                    .unwrap();
                let net = network(Value::GpuTensor(weights.clone()), Value::Num(1.0));
                provider.reset_telemetry();
                let error = futures::executor::block_on(admit_export_call(
                    crate::builtins::common::test_support::builtin_arguments(vec![
                        net,
                        Value::String("unused.onnx".into()),
                        Value::String(option.into()),
                        Value::GpuTensor(control.clone()),
                    ]),
                ))
                .unwrap_err();
                assert!(
                    error.message.contains("scalar storage"),
                    "{option}: {error}"
                );
                assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
                for retained in [control, weights] {
                    let actual =
                        crate::builtins::common::test_support::gather(Value::GpuTensor(retained))
                            .unwrap();
                    assert_eq!(actual.materialize_f64(), vec![1.0, 2.0]);
                }
            }
        });
    }

    #[test]
    fn resident_name_control_uses_text_admission_not_numeric_extension_policy() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let name = provider
                .upload(&HostTensorView {
                    data: &[1.0],
                    shape: &[1, 1],
                })
                .unwrap();
            let weights = provider
                .upload(&HostTensorView {
                    data: &[1.0, 2.0],
                    shape: &[1, 2],
                })
                .unwrap();
            provider.reset_telemetry();
            let error = futures::executor::block_on(admit_export_call(
                crate::builtins::common::test_support::builtin_arguments(vec![
                    network(Value::GpuTensor(weights.clone()), Value::Num(1.0)),
                    Value::String("unused.onnx".into()),
                    Value::String("InputNames".into()),
                    Value::GpuTensor(name.clone()),
                ]),
            ))
            .unwrap_err();
            assert!(error.message.contains("InputNames"), "{error}");
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
            name.allocation_lease().validate_live().unwrap();
            weights.allocation_lease().validate_live().unwrap();
        });
    }

    #[test]
    fn unknown_and_known_invalid_options_precede_learnable_download() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let weights = provider
                .upload(&HostTensorView {
                    data: &[1.0, 2.0],
                    shape: &[1, 2],
                })
                .unwrap();
            for (name, value) in [
                ("UnknownOption", Value::Num(1.0)),
                ("OpsetVersion", Value::Bool(true)),
                (
                    "InputNames",
                    Value::StringArray(
                        runmat_value::StringArray::new(vec!["a".into(), "b".into()], vec![1, 2])
                            .unwrap(),
                    ),
                ),
            ] {
                provider.reset_telemetry();
                let error = futures::executor::block_on(admit_export_call(
                    crate::builtins::common::test_support::builtin_arguments(vec![
                        network(Value::GpuTensor(weights.clone()), Value::Num(1.0)),
                        Value::String("unused.onnx".into()),
                        Value::String(name.into()),
                        value,
                    ]),
                ))
                .unwrap_err();
                assert!(
                    error.message.contains(name)
                        || error.message.contains(&name.to_ascii_lowercase()),
                    "{name}: {error}"
                );
                assert_eq!(provider.telemetry_snapshot().download_bytes, 0, "{name}");
            }
            weights.allocation_lease().validate_live().unwrap();
        });
    }

    #[test]
    fn public_option_names_fold_case_and_duplicate_values_before_typed_admission() {
        let parsed = parse_option_inputs(crate::builtins::common::test_support::builtin_arguments(
            vec![
                Value::String("OpsetVersion".into()),
                Value::Num(13.0),
                Value::String("OPSETVERSION".into()),
                Value::Num(14.0),
                Value::String("InputNames".into()),
                Value::String("input".into()),
            ],
        ))
        .unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].key, Some(ExportOnnxOptionKey::InputNames));
        assert_eq!(parsed[1].key, Some(ExportOnnxOptionKey::OpsetVersion));
        assert!(matches!(&parsed[1].value, Value::Num(number) if *number == 14.0));
    }

    #[test]
    fn verbose_accepts_scalar_host_logical_storage() {
        let options = parse_export_options(std::collections::BTreeMap::from([(
            ExportOnnxOptionKey::Verbose,
            Value::LogicalArray(runmat_value::LogicalArray::new(vec![1], vec![1, 1]).unwrap()),
        )]))
        .unwrap();
        assert_eq!(
            options.opset_version,
            ExportOptions::default().opset_version
        );
        let nonscalar =
            Value::LogicalArray(runmat_value::LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap());
        assert!(validate_option(ExportOnnxOptionKey::Verbose, &nonscalar).is_err());
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
                    .replace_dl_property("State", Value::GpuTensor(unrelated))
                    .expect("fixture property");
            }
            provider.reset_telemetry();
            let admitted = futures::executor::block_on(admit_export_call(
                crate::builtins::common::test_support::builtin_arguments(vec![
                    net,
                    Value::String("unused.onnx".into()),
                ]),
            ))
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
        let error = futures::executor::block_on(admit_export_call(
            crate::builtins::common::test_support::builtin_arguments(vec![
                network(weights, Value::Num(1.0)),
                Value::String("unused.onnx".into()),
                Value::String("OpsetVersion".into()),
                opset,
            ]),
        ))
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:ExportOnnxIntegerControlExtension")
        );
    }
}
