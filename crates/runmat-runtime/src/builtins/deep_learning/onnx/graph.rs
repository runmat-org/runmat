use super::*;

fn onnx_matrix_dimensions(tensor: &Tensor) -> (usize, usize) {
    match tensor.shape() {
        [] => (0, 0),
        [columns] => (1, *columns),
        [rows, columns, ..] => (*rows, *columns),
    }
}

pub(super) fn build_export_graph(
    network: &ObjectInstance,
    layers: &[Value],
    options: &ExportOptions,
) -> BuiltinResult<ExportGraph> {
    if layers.is_empty() {
        return Err(deep_learning_error(
            EXPORT_ONNX,
            "exportONNXNetwork: network must contain at least one layer",
        ));
    }

    let layer_names = super::layer_names(layers, EXPORT_ONNX)?;
    ensure_unique_layer_names(&layer_names)?;
    let input_name = match &options.input_name {
        Some(name) => name.clone(),
        None => network_name(network, "InputNames")?.unwrap_or_else(|| layer_names[0].clone()),
    };
    let output_name = match &options.output_name {
        Some(name) => name.clone(),
        None => network_name(network, "OutputNames")?.unwrap_or_else(|| "output".to_string()),
    };

    let mut nodes = Vec::new();
    let mut initializers = Vec::new();
    let mut current_tensor = input_name.clone();
    let mut current_features = None;
    let mut saw_input = false;
    let mut export_dtype = None;

    for (idx, layer_value) in layers.iter().enumerate() {
        let Value::Object(layer) = layer_value else {
            return Err(deep_learning_error(
                EXPORT_ONNX,
                "exportONNXNetwork: network layers must be layer objects",
            ));
        };
        let layer_name = layer_names[idx].clone();
        match ForwardLayerKind::from_identity(layer.class_name()) {
            Some(ForwardLayerKind::FeatureInput) => {
                if idx != 0 {
                    return Err(deep_learning_error(
                        EXPORT_ONNX,
                        "exportONNXNetwork: featureInputLayer must be the first exported layer",
                    ));
                }
                current_features = Some(model::feature_input_width(layer, EXPORT_ONNX)? as i64);
                saw_input = true;
            }
            Some(ForwardLayerKind::FullyConnected) => {
                let input_features = current_features.ok_or_else(|| {
                    deep_learning_error(
                        EXPORT_ONNX,
                        "exportONNXNetwork: fullyConnectedLayer requires a preceding featureInputLayer",
                    )
                })?;
                let weights = model::tensor_property(layer, "Weights", EXPORT_ONNX)?;
                let (weight_rows, weight_cols) = onnx_matrix_dimensions(&weights);
                if weights.shape().len() > 2 || weight_cols as i64 != input_features {
                    return Err(deep_learning_error(
                        EXPORT_ONNX,
                        format!(
                            "exportONNXNetwork: fullyConnectedLayer Weights must be outputSize-by-{input_features}"
                        ),
                    ));
                }
                let bias = model::tensor_property(layer, "Bias", EXPORT_ONNX)?;
                if bias.len() != weight_rows {
                    return Err(deep_learning_error(
                        EXPORT_ONNX,
                        format!(
                            "exportONNXNetwork: fullyConnectedLayer Bias must have {} elements",
                            weight_rows
                        ),
                    ));
                }
                let output_size =
                    model::positive_property_usize(layer, "OutputSize", EXPORT_ONNX)? as i64;
                if output_size as usize != weight_rows {
                    return Err(deep_learning_error(
                        EXPORT_ONNX,
                        "exportONNXNetwork: fullyConnectedLayer OutputSize must match Weights rows",
                    ));
                }
                let weight_name = format!("{layer_name}.Weights");
                let bias_name = format!("{layer_name}.Bias");
                export_dtype = Some(merge_export_dtype(export_dtype, weights.numeric_dtype()));
                export_dtype = Some(merge_export_dtype(export_dtype, bias.numeric_dtype()));
                initializers.push(Initializer {
                    name: weight_name.clone(),
                    dims: vec![weight_rows as i64, weight_cols as i64],
                    data: tensor_to_row_major_2d(weights)?,
                });
                initializers.push(Initializer {
                    name: bias_name.clone(),
                    dims: vec![weight_rows as i64],
                    data: tensor_storage(bias)?,
                });
                let next_tensor = format!("{layer_name}/Gemm");
                nodes.push(Node {
                    name: layer_name,
                    op_type: "Gemm".to_string(),
                    inputs: vec![current_tensor, weight_name, bias_name],
                    outputs: vec![next_tensor.clone()],
                    attributes: vec![Attribute::Int {
                        name: "transB",
                        value: 1,
                    }],
                });
                current_tensor = next_tensor;
                current_features = Some(output_size);
            }
            Some(ForwardLayerKind::Relu) => {
                let next_tensor = format!("{layer_name}/Relu");
                nodes.push(Node {
                    name: layer_name,
                    op_type: "Relu".to_string(),
                    inputs: vec![current_tensor],
                    outputs: vec![next_tensor.clone()],
                    attributes: vec![],
                });
                current_tensor = next_tensor;
            }
            Some(ForwardLayerKind::Elu) => {
                let alpha = layer
                    .properties()
                    .get("Alpha")
                    .map(|value| super::numeric_scalar(value, EXPORT_ONNX, "Alpha"))
                    .transpose()?
                    .unwrap_or(1.0);
                let next_tensor = format!("{layer_name}/Elu");
                nodes.push(Node {
                    name: layer_name,
                    op_type: "Elu".to_string(),
                    inputs: vec![current_tensor],
                    outputs: vec![next_tensor.clone()],
                    attributes: vec![Attribute::Float {
                        name: "alpha",
                        value: alpha as f32,
                    }],
                });
                current_tensor = next_tensor;
            }
            Some(ForwardLayerKind::Softmax) => {
                let next_tensor = format!("{layer_name}/Softmax");
                nodes.push(Node {
                    name: layer_name,
                    op_type: "Softmax".to_string(),
                    inputs: vec![current_tensor],
                    outputs: vec![next_tensor.clone()],
                    attributes: vec![Attribute::Int {
                        name: "axis",
                        value: 1,
                    }],
                });
                current_tensor = next_tensor;
            }
            Some(kind) if kind.is_output() => {
                if idx + 1 != layers.len() {
                    return Err(deep_learning_error(
                        EXPORT_ONNX,
                        "exportONNXNetwork: output layers must be terminal layers",
                    ));
                }
            }
            _ => {
                return Err(unsupported_error(
                    EXPORT_ONNX,
                    format!(
                        "exportONNXNetwork: layer type '{}' cannot be exported to ONNX",
                        layer.class_name()
                    ),
                ));
            }
        }
    }

    if !saw_input {
        return Err(deep_learning_error(
            EXPORT_ONNX,
            "exportONNXNetwork: network must start with a supported input layer",
        ));
    }
    let output_features = current_features.ok_or_else(|| {
        deep_learning_error(
            EXPORT_ONNX,
            "exportONNXNetwork: exported graph must produce a numeric feature tensor",
        )
    })?;
    if current_tensor != output_name {
        nodes.push(Node {
            name: "output".into(),
            op_type: "Identity".to_string(),
            inputs: vec![current_tensor],
            outputs: vec![output_name.clone()],
            attributes: vec![],
        });
    }

    Ok(ExportGraph {
        nodes,
        initializers,
        input: ValueInfo {
            name: input_name,
            batch_size: options.batch_size,
            features: feature_input_width_from_first_layer(layers)?,
        },
        output: ValueInfo {
            name: output_name,
            batch_size: options.batch_size,
            features: output_features,
        },
        dtype: export_dtype.unwrap_or(NumericDType::F64),
    })
}

fn feature_input_width_from_first_layer(layers: &[Value]) -> BuiltinResult<i64> {
    let Some(Value::Object(layer)) = layers.first() else {
        return Err(deep_learning_error(
            EXPORT_ONNX,
            "exportONNXNetwork: network must start with a featureInputLayer",
        ));
    };
    if !layer.class_name().is(super::FEATURE_INPUT_LAYER_CLASS) {
        return Err(deep_learning_error(
            EXPORT_ONNX,
            "exportONNXNetwork: network must start with a featureInputLayer",
        ));
    }
    Ok(model::feature_input_width(layer, EXPORT_ONNX)? as i64)
}

fn ensure_unique_layer_names(names: &[String]) -> BuiltinResult<()> {
    let mut seen = BTreeSet::new();
    for name in names {
        if !seen.insert(name.clone()) {
            return Err(deep_learning_error(
                EXPORT_ONNX,
                format!("exportONNXNetwork: duplicate layer name '{name}'"),
            ));
        }
    }
    Ok(())
}

pub(super) fn network_name(
    network: &ObjectInstance,
    property: &str,
) -> BuiltinResult<Option<String>> {
    let Some(value) = network.properties().get(property) else {
        return Ok(None);
    };
    match value {
        Value::StringArray(array) if array.data().is_empty() => Ok(None),
        Value::StringArray(array) if array.data().len() == 1 && array.data()[0].as_str().is_some_and(|text| !text.is_empty()) => {
            Ok(Some(array.data()[0].text_ref().expect("checked nonmissing scalar string").to_string()))
        }
        Value::StringArray(_) => Err(deep_learning_error(
            EXPORT_ONNX,
            format!("exportONNXNetwork: network {property} metadata must contain exactly one non-empty name"),
        )),
        Value::String(runmat_value::StringElement::Text(name)) if !name.is_empty() => Ok(Some(name.to_string())),
        Value::String(_) => Ok(None),
        _ => Err(deep_learning_error(
            EXPORT_ONNX,
            format!("exportONNXNetwork: network {property} metadata must be text"),
        )),
    }
}

pub(super) fn merge_export_dtype(
    current: Option<NumericDType>,
    dtype: NumericDType,
) -> NumericDType {
    if current == Some(NumericDType::F64) || dtype == NumericDType::F64 {
        NumericDType::F64
    } else {
        NumericDType::F32
    }
}

fn tensor_storage(tensor: Tensor) -> BuiltinResult<NumericStorage> {
    tensor
        .into_numeric_storage()
        .map_err(|err| deep_learning_error(EXPORT_ONNX, format!("exportONNXNetwork: {err}")))
}

pub(super) fn tensor_to_row_major_2d(tensor: Tensor) -> BuiltinResult<NumericStorage> {
    let (rows, cols) = onnx_matrix_dimensions(&tensor);
    let mut indices = Vec::with_capacity(rows * cols);
    for row in 0..rows {
        for col in 0..cols {
            indices.push(row + col * rows);
        }
    }
    tensor_storage(tensor)?
        .reorder(&indices)
        .map_err(|err| deep_learning_error(EXPORT_ONNX, format!("exportONNXNetwork: {err}")))
}
