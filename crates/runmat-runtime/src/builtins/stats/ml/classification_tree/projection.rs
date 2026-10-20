use super::*;

pub(super) fn model_object(
    runtime: Option<&crate::context::RuntimeContext>,
    spec: FitSpec,
    fit: TreeFit,
) -> BuiltinResult<Value> {
    let (observation_count, predictor_count) = classification_tree_matrix_dimensions(&spec.x);
    let response_name = Value::string(spec.response_name);
    let predictor_names = string_column(&spec.predictor_names)?;
    let class_names = labels_value(&spec.class_names, spec.class_kind)?;
    let categorical = Value::Tensor(Tensor::new(Vec::new(), vec![0, 1]).map_err(|err| {
        fitctree_internal(format!(
            "fitctree: categorical predictor shape error: {err}"
        ))
    })?);
    let mut params = runmat_value::UnboundedStructValueBuilder::new();
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
    let model_parameters = Value::Struct(params.finish());
    let class_kind = Value::string(
        match spec.class_kind {
            LabelKind::Numeric => "numeric",
            LabelKind::Integer(class) => integer_class_name(class),
            LabelKind::Text => "text",
            LabelKind::Logical => "logical",
        }
        .to_string(),
    );
    let fields = [
        ("ResponseName", response_name),
        ("PredictorNames", predictor_names),
        ("ClassNames", class_names),
        ("CategoricalPredictors", categorical),
        ("ScoreTransform", Value::string("none".to_string())),
        ("NumObservations", Value::Num(observation_count as f64)),
        ("NumPredictors", Value::Num(predictor_count as f64)),
        ("NumNodes", Value::Num(fit.nodes.len() as f64)),
        ("NumSplits", Value::Num(fit.num_splits as f64)),
        ("ModelParameters", model_parameters),
        ("__RunMatTreeClassKind", class_kind),
        (
            "__RunMatTreeClassNames",
            labels_value(&spec.class_names, spec.class_kind)?,
        ),
        ("__RunMatTreeChildren", tree_children_tensor(&fit.nodes)?),
        ("__RunMatTreePredictor", tree_predictor_tensor(&fit.nodes)?),
        ("__RunMatTreeThreshold", tree_threshold_tensor(&fit.nodes)?),
        ("__RunMatTreeNodeClass", tree_class_tensor(&fit.nodes)?),
        (
            "__RunMatTreeNodeProb",
            tree_prob_tensor(&fit.nodes, spec.class_names.len())?,
        ),
    ];
    let object = crate::execution::object::construct_object_from_borrowed_fields(
        runtime,
        runmat_types::standard::CLASSIFICATION_TREE.display_name(),
        fields.iter().map(|(name, value)| (*name, value)),
    )?;
    Ok(object.into_value_and_lease().0)
}
