use super::*;

pub(crate) fn predict_classification_tree_object(
    object: ObjectInstance,
    xnew: Value,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<Vec<Value>> {
    if !rest.is_empty() {
        return Err(predict_invalid(
            "predict: ClassificationTree Subtrees and name-value options are not supported yet",
        ));
    }
    let predictor_names = string_property(&object, "PredictorNames")?;
    let x = predictors_for_prediction(xnew, &predictor_names)?;
    let children = matrix_property(&object, "__RunMatTreeChildren")?;
    let predictors = numeric_property(&object, "__RunMatTreePredictor")?;
    let thresholds = numeric_property(&object, "__RunMatTreeThreshold")?;
    let node_classes = numeric_property(&object, "__RunMatTreeNodeClass")?;
    let node_probs = matrix_property(&object, "__RunMatTreeNodeProb")?;
    let class_names_value = object
        .properties()
        .get("__RunMatTreeClassNames")
        .cloned()
        .ok_or_else(|| predict_invalid("predict: tree is missing class names"))?;
    let class_kind = class_kind_property(&object)?;
    let class_names = labels_from_value(class_names_value, "ClassNames")?;
    let class_count = class_names.labels.len();
    let node_count = predictors.len();
    let (x_rows, _) = classification_tree_matrix_dimensions(&x);
    if children.len() != node_count * 2
        || thresholds.len() != node_count
        || node_classes.len() != node_count
        || node_probs.len() != node_count * class_count
    {
        return Err(predict_invalid("predict: tree metadata is inconsistent"));
    }
    let mut label_indices = Vec::with_capacity(x_rows);
    let mut score_data = vec![0.0; x_rows * class_count];
    let mut node_data = Vec::with_capacity(x_rows);
    let mut class_number_data = Vec::with_capacity(x_rows);
    for row in 0..x_rows {
        let node = traverse_tree(&x, row, &children, &predictors, &thresholds, node_count)?;
        let class_index = checked_one_based_index(node_classes[node], class_count, "class number")?;
        label_indices.push(class_index);
        node_data.push(node as f64 + 1.0);
        class_number_data.push(class_index as f64 + 1.0);
        for class in 0..class_count {
            score_data[class * x_rows + row] = node_probs[class * node_count + node];
        }
    }
    let labels = predicted_labels_value(&class_names.labels, class_kind, &label_indices)?;
    let scores = Value::Tensor(
        Tensor::new(score_data, vec![x_rows, class_count])
            .map_err(|err| predict_internal(format!("predict: {err}")))?,
    );
    let node = Value::Tensor(
        Tensor::new(node_data, vec![x_rows, 1])
            .map_err(|err| predict_internal(format!("predict: {err}")))?,
    );
    let cnum = Value::Tensor(
        Tensor::new(class_number_data, vec![x_rows, 1])
            .map_err(|err| predict_internal(format!("predict: {err}")))?,
    );
    Ok(vec![labels, scores, node, cnum])
}

fn traverse_tree(
    x: &Tensor,
    row: usize,
    children: &[f64],
    predictors: &[f64],
    thresholds: &[f64],
    node_count: usize,
) -> BuiltinResult<usize> {
    let mut node = 0usize;
    for _ in 0..node_count {
        let predictor = predictors[node];
        if predictor <= 0.0 {
            return Ok(node);
        }
        let predictor_idx = checked_one_based_index(
            predictor,
            classification_tree_matrix_dimensions(x).1,
            "predictor",
        )?;
        let value = x_value(x, row, predictor_idx);
        if value.is_nan() {
            return Ok(node);
        }
        let child_col = if value <= thresholds[node] { 0 } else { 1 };
        let child = children[child_col * node_count + node];
        if child <= 0.0 {
            return Ok(node);
        }
        node = checked_one_based_index(child, node_count, "child node")?;
    }
    Err(predict_invalid("predict: tree traversal did not terminate"))
}

fn checked_one_based_index(value: f64, len: usize, label: &str) -> BuiltinResult<usize> {
    if !value.is_finite() || value < 1.0 || value.fract() != 0.0 {
        return Err(predict_invalid(format!("predict: invalid {label}")));
    }
    let index = value as usize - 1;
    if index >= len {
        return Err(predict_invalid(format!("predict: {label} out of range")));
    }
    Ok(index)
}

fn predictors_for_prediction(value: Value, predictor_names: &[String]) -> BuiltinResult<Tensor> {
    if let Value::Object(object) = &value {
        if is_tabular_object(object) {
            let variables = table_variables(object)
                .map_err(|err| predict_invalid(format!("predict: {err}")))?;
            let mut columns = Vec::with_capacity(predictor_names.len());
            for name in predictor_names {
                let raw = variables.fields.get(name).ok_or_else(|| {
                    predict_invalid(format!("predict: missing predictor '{name}'"))
                })?;
                let tensor =
                    tensor::value_into_tensor_for(PREDICT_NAME, raw.clone()).map_err(|_| {
                        predict_invalid(format!("predict: predictor '{name}' must be numeric"))
                    })?;
                ensure_integer_tensor_exact_predict(&tensor, name)?;
                columns.push(vector_values_predict(&tensor, name)?);
            }
            return columns_to_tensor_predict(columns);
        }
    }
    let tensor = tensor::value_into_tensor_for(PREDICT_NAME, value)
        .map_err(|err| predict_invalid(format!("predict: {err}")))?;
    ensure_integer_tensor_exact_predict(&tensor, "predictors")?;
    if tensor.shape().len() > 2
        || classification_tree_matrix_dimensions(&tensor).1 != predictor_names.len()
    {
        return Err(predict_invalid(format!(
            "predict: X must have {} predictor columns",
            predictor_names.len()
        )));
    }
    Ok(tensor)
}
