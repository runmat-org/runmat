use super::*;

pub(super) fn take_training_layers(
    value: Value,
    function: &'static str,
) -> BuiltinResult<Vec<Value>> {
    match value {
        Value::Object(mut object)
            if object.class_name.is(super::super::LAYER_GRAPH_CLASS)
                || model::is_deep_learning_network_object(&object) =>
        {
            let layers = object
                .properties
                .remove("Layers")
                .expect("nonempty Layers collection was admitted above");
            model::layers_from_network_value(layers, function)
        }
        other => model::layers_from_network_value(other, function),
    }
}

pub(super) fn admit_training_output_count(function: &'static str) -> BuiltinResult<()> {
    if let Some(requested) = crate::output_count::current_output_count() {
        if requested > 2 {
            return Err(deep_learning_error(
                function,
                format!("{function}: requested {requested} outputs, but at most 2 are supported"),
            ));
        }
    }
    Ok(())
}

pub(super) fn admit_training_observations(
    observations: Option<usize>,
    function: &'static str,
) -> BuiltinResult<()> {
    if observations == Some(0) {
        return Err(deep_learning_error(
            function,
            format!("{function}: X must contain at least one observation"),
        ));
    }
    Ok(())
}

pub(super) fn admit_training_cancellation(function: &'static str) -> BuiltinResult<()> {
    if crate::interrupt::is_cancelled() {
        return Err(crate::runtime_error::semantic_error(
            "RunMat:Cancelled",
            format!("{function}: training was cancelled"),
        ));
    }
    Ok(())
}

pub(super) fn resident_rows(value: &Value, function: &'static str) -> BuiltinResult<Option<usize>> {
    let Value::GpuTensor(handle) = value else {
        return Ok(None);
    };
    if handle.shape().len() > 2
        || runmat_accelerate_api::handle_storage(handle)
            == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
    {
        return Err(deep_learning_error(
            function,
            format!("{function}: X must be a real 2-D numeric or logical matrix"),
        ));
    }
    let rows = match handle.shape() {
        [] => 0,
        [_] => 1,
        [rows, _] => *rows,
        _ => unreachable!("rank was checked above"),
    };
    Ok(Some(rows))
}

pub(super) fn host_matrix(
    value: &Value,
    function: &'static str,
    label: &str,
) -> BuiltinResult<Option<Matrix>> {
    if matches!(value, Value::GpuTensor(_)) {
        Ok(None)
    } else {
        numeric_matrix_ref(value, function, label).map(Some)
    }
}

pub(super) async fn materialize_matrix(
    value: &Value,
    admitted: Option<Matrix>,
    function: &'static str,
    label: &str,
) -> BuiltinResult<Matrix> {
    match admitted {
        Some(matrix) => Ok(matrix),
        None => numeric_matrix(crate::gather_if_needed_async(value).await?, function, label),
    }
}

fn preflight_host_learnable_values(
    value: &Value,
    function: &'static str,
    property: &str,
) -> BuiltinResult<()> {
    let data = match value {
        Value::Object(object) if object.class_name.is(super::super::DLARRAY_CLASS) => {
            object.properties.get("Data").ok_or_else(|| {
                deep_learning_error(
                    function,
                    format!("{function}: dlarray {property} is missing Data"),
                )
            })?
        }
        other => other,
    };
    let finite = match data {
        Value::GpuTensor(_) => true,
        Value::Tensor(tensor) => (0..tensor.len()).all(|index| {
            tensor
                .numeric_value_at(index)
                .is_some_and(runmat_value::NumericScalar::is_finite)
        }),
        Value::Num(value) => value.is_finite(),
        Value::Int(_) | Value::Bool(_) => true,
        _ => false,
    };
    if !finite {
        return Err(deep_learning_error(
            function,
            format!("{function}: {property} must contain finite real numeric values"),
        ));
    }
    Ok(())
}

pub(super) fn preflight_training_layers(
    layers: &[Value],
    function: &'static str,
) -> BuiltinResult<()> {
    let mut width = None;
    for layer in layers {
        let Value::Object(object) = layer else {
            return Err(deep_learning_error(
                function,
                format!("{function}: network layers must be objects"),
            ));
        };
        match ForwardLayerKind::from_identity(&object.class_name) {
            Some(ForwardLayerKind::FeatureInput) => {
                width = Some(model::feature_input_width(object, function)?)
            }
            Some(ForwardLayerKind::FullyConnected) => {
                let input_width = width.ok_or_else(|| {
                    deep_learning_error(
                        function,
                        format!("{function}: fullyConnectedLayer requires preceding feature input"),
                    )
                })?;
                let output_size =
                    model::fully_connected_output_width(object, input_width, function)?;
                for property in ["Weights", "Bias"] {
                    let value = object
                        .properties
                        .get(property)
                        .expect("shared FC preflight verified learnable property");
                    preflight_host_learnable_values(value, function, property)?;
                }
                width = Some(output_size);
            }
            Some(ForwardLayerKind::Elu) => {
                if let Some(alpha) = object.properties.get("Alpha") {
                    let _ = super::super::numeric_scalar(alpha, function, "Alpha")?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}

pub(super) async fn gather_training_learnables(layers: &mut [Value]) -> BuiltinResult<()> {
    for layer in layers {
        let Value::Object(object) = layer else {
            continue;
        };
        if !object
            .class_name
            .is(super::super::FULLY_CONNECTED_LAYER_CLASS)
        {
            continue;
        }
        for property in ["Weights", "Bias"] {
            let value = object
                .properties
                .remove(property)
                .expect("preflight verified learnable");
            let prepared = match value {
                Value::Object(mut dlarray)
                    if dlarray.class_name.is(super::super::DLARRAY_CLASS) =>
                {
                    let data = dlarray
                        .properties
                        .remove("Data")
                        .expect("preflight verified dlarray Data");
                    let gathered = match data {
                        Value::GpuTensor(_) => crate::gather_if_needed_async(&data).await?,
                        host => host,
                    };
                    dlarray.properties.insert("Data".to_string(), gathered);
                    Value::Object(dlarray)
                }
                Value::GpuTensor(_) => crate::gather_if_needed_async(&value).await?,
                host => host,
            };
            object.properties.insert(property.to_string(), prepared);
        }
    }
    Ok(())
}
