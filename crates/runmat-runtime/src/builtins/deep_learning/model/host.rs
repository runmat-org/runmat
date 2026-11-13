//! Output metadata and host parameter layout adaptation.
use super::*;

pub(super) fn dlarray_format(value: &Value, function: &'static str) -> BuiltinResult<String> {
    let Value::Object(object) = value else {
        return Ok(String::new());
    };
    if !object.class_name().is(super::DLARRAY_CLASS) {
        return Ok(String::new());
    }
    let format = match object.properties().get("Format") {
        None => "",
        Some(Value::String(runmat_value::StringElement::Text(text))) => text.as_str(),
        Some(_) => {
            return Err(deep_learning_error(
                function,
                format!("{function}: dlarray Format must be text"),
            ))
        }
    };
    let labels = match object.properties().get("Labels") {
        None => format,
        Some(Value::String(runmat_value::StringElement::Text(text))) => text.as_str(),
        Some(_) => {
            return Err(deep_learning_error(
                function,
                format!("{function}: dlarray Labels must be text"),
            ))
        }
    };
    if labels != format {
        return Err(deep_learning_error(
            function,
            format!("{function}: dlarray Format and Labels disagree"),
        ));
    }
    Ok(format.to_owned())
}

pub(super) fn elu_alpha(layer: &ObjectInstance, function: &'static str) -> BuiltinResult<f64> {
    Ok(layer
        .properties()
        .get("Alpha")
        .map(|value| super::super::numeric_scalar(value, function, "Alpha"))
        .transpose()?
        .unwrap_or(1.0))
}

fn host_matrix_dimensions(tensor: &Tensor) -> (usize, usize) {
    match tensor.shape() {
        [] => (0, 0),
        [columns] => (1, *columns),
        [rows, columns, ..] => (*rows, *columns),
    }
}

pub(super) fn transpose_2d(tensor: &Tensor, function: &'static str) -> BuiltinResult<Tensor> {
    if tensor.shape().len() > 2 {
        return Err(deep_learning_error(
            function,
            "predict: ObservationsIn='columns' requires a 2-D matrix",
        ));
    }
    let (rows, cols) = host_matrix_dimensions(tensor);
    let mut indices = vec![0usize; tensor::tensor_element_len(tensor)];
    for row in 0..rows {
        for col in 0..cols {
            indices[col + row * cols] = row + col * rows;
        }
    }
    let storage = tensor
        .clone()
        .into_numeric_storage()
        .map_err(|err| deep_learning_error(function, err))?
        .reorder(&indices)
        .map_err(|err| deep_learning_error(function, err))?;
    Tensor::from_numeric_storage(storage, vec![cols, rows])
        .map_err(|err| deep_learning_error(function, err))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dlarray_text_metadata_defaults_only_when_properties_are_absent() {
        use runmat_value::StringElement;
        for format in [
            None,
            Some(Value::from("")),
            Some(Value::from("CB")),
            Some(Value::from("<missing>")),
            Some(Value::String(StringElement::Missing)),
            Some(Value::Bool(false)),
        ] {
            for labels in [
                None,
                Some(Value::from("")),
                Some(Value::from("CB")),
                Some(Value::from("<missing>")),
                Some(Value::String(StringElement::Missing)),
                Some(Value::Bool(false)),
            ] {
                let mut object = ObjectInstance::new(super::super::DLARRAY_CLASS.to_owned());
                if let Some(format) = &format {
                    object.replace_property_unbounded("Format".into(), format.clone());
                }
                if let Some(labels) = &labels {
                    object.replace_property_unbounded("Labels".into(), labels.clone());
                }
                let expected_format = match &format {
                    None => Some(""),
                    Some(Value::String(text)) => text.as_str(),
                    _ => None,
                };
                let expected_labels = match &labels {
                    None => expected_format,
                    Some(Value::String(text)) => text.as_str(),
                    _ => None,
                };
                let result = dlarray_format(&Value::Object(object), "forward");
                match (expected_format, expected_labels) {
                    (Some(format), Some(labels)) if format == labels => {
                        assert_eq!(result.unwrap(), format)
                    }
                    _ => assert!(result.is_err()),
                }
            }
        }
        assert_eq!(dlarray_format(&Value::Num(1.0), "forward").unwrap(), "");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn host_transpose_preserves_dense_rank_one_matrix_geometry() {
        let row = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let transposed = transpose_2d(&row, "predict").expect("rank-one row transpose");
        assert_eq!(transposed.shape(), [3, 1]);
        assert_eq!(transposed.materialize_f64(), vec![1.0, 2.0, 3.0]);

        let scalar = Tensor::new(vec![1.0], Vec::new()).unwrap();
        assert_eq!(host_matrix_dimensions(&scalar), (0, 0));
        assert!(transpose_2d(&scalar, "predict").is_err());
    }
}
