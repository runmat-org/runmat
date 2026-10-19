use super::*;

pub(super) fn model_object(spec: FitSpec, fit: TreeFit) -> BuiltinResult<ObjectInstance> {
    let (observation_count, predictor_count) = classification_tree_matrix_dimensions(&spec.x);
    let mut object = ObjectInstance::new(runmat_types::standard::CLASSIFICATION_TREE.to_string());
    object.properties.insert(
        "ResponseName".to_string(),
        Value::string(spec.response_name),
    );
    object.properties.insert(
        "PredictorNames".to_string(),
        string_column(&spec.predictor_names)?,
    );
    object.properties.insert(
        "ClassNames".to_string(),
        labels_value(&spec.class_names, spec.class_kind)?,
    );
    object.properties.insert(
        "CategoricalPredictors".to_string(),
        Value::Tensor(Tensor::new(Vec::new(), vec![0, 1]).map_err(|err| {
            fitctree_internal(format!(
                "fitctree: categorical predictor shape error: {err}"
            ))
        })?),
    );
    object.properties.insert(
        "ScoreTransform".to_string(),
        Value::string("none".to_string()),
    );
    object.properties.insert(
        "NumObservations".to_string(),
        Value::Num(observation_count as f64),
    );
    object.properties.insert(
        "NumPredictors".to_string(),
        Value::Num(predictor_count as f64),
    );
    object
        .properties
        .insert("NumNodes".to_string(), Value::Num(fit.nodes.len() as f64));
    object
        .properties
        .insert("NumSplits".to_string(), Value::Num(fit.num_splits as f64));
    let mut params = StructValue::new();
    params.insert(
        "MaxNumSplits",
        Value::Num(spec.options.max_num_splits.unwrap_or(0) as f64),
    );
    params.insert("MinLeaf", Value::Num(spec.options.min_leaf_size as f64));
    params.insert("MinParent", Value::Num(spec.options.min_parent_size as f64));
    params.insert(
        "SplitCriterion",
        Value::string(
            match spec.options.split_criterion {
                SplitCriterion::Gdi => "gdi",
                SplitCriterion::Deviance => "deviance",
            }
            .to_string(),
        ),
    );
    object
        .properties
        .insert("ModelParameters".to_string(), Value::Struct(params));
    object.properties.insert(
        "__RunMatTreeClassKind".to_string(),
        Value::string(
            match spec.class_kind {
                LabelKind::Numeric => "numeric",
                LabelKind::Integer(class) => integer_class_name(class),
                LabelKind::Text => "text",
                LabelKind::Logical => "logical",
            }
            .to_string(),
        ),
    );
    object.properties.insert(
        "__RunMatTreeClassNames".to_string(),
        labels_value(&spec.class_names, spec.class_kind)?,
    );
    object.properties.insert(
        "__RunMatTreeChildren".to_string(),
        tree_children_tensor(&fit.nodes)?,
    );
    object.properties.insert(
        "__RunMatTreePredictor".to_string(),
        tree_predictor_tensor(&fit.nodes)?,
    );
    object.properties.insert(
        "__RunMatTreeThreshold".to_string(),
        tree_threshold_tensor(&fit.nodes)?,
    );
    object.properties.insert(
        "__RunMatTreeNodeClass".to_string(),
        tree_class_tensor(&fit.nodes)?,
    );
    object.properties.insert(
        "__RunMatTreeNodeProb".to_string(),
        tree_prob_tensor(&fit.nodes, spec.class_names.len())?,
    );
    Ok(object)
}
