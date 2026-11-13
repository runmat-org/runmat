use super::*;
pub(super) fn confusionmat_compute(
    group: Value,
    grouphat: Value,
    rest: crate::call::FunctionArgumentView<'_>,
    output_limit: usize,
) -> BuiltinResult<Vec<Value>> {
    let group = labels_from_value(group, "group")?;
    let grouphat = labels_from_value(grouphat, "grouphat")?;
    if group.labels.len() != grouphat.labels.len() {
        return Err(invalid(
            "confusionmat: group and grouphat must contain the same number of labels",
        ));
    }
    if !label_kinds_compatible(group.kind, grouphat.kind) {
        return Err(invalid(
            "confusionmat: group and grouphat label types must be compatible",
        ));
    }
    let order = parse_options(rest)?;
    let (order, order_kind) = if let Some(order) = order {
        if !label_kinds_compatible(group.kind, order.kind) {
            return Err(invalid(
                "confusionmat: Order labels must match group label type",
            ));
        }
        let order_kind = order.kind;
        let order = dedupe_labels(order.labels);
        ensure_order_covers(&order, &group.labels, &grouphat.labels)?;
        (order, order_kind)
    } else {
        (inferred_order(&group, &grouphat), group.kind)
    };
    let class_count = order.len();
    let cell_count = class_count
        .checked_mul(class_count)
        .ok_or_else(|| invalid("confusionmat: too many classes"))?;
    if cell_count > MAX_CONFUSIONMAT_CELLS {
        return Err(invalid("confusionmat: too many classes"));
    }
    if output_limit == 0 {
        return Ok(Vec::new());
    }
    let mut index = HashMap::with_capacity(order.len());
    for (idx, label) in order.iter().enumerate() {
        index.insert(label_key(label), idx);
    }
    let mut counts = vec![0.0; cell_count];
    for (actual, predicted) in group.labels.iter().zip(grouphat.labels.iter()) {
        if is_missing_label(actual) || is_missing_label(predicted) {
            continue;
        }
        let row = *index
            .get(&label_key(actual))
            .ok_or_else(|| internal("confusionmat: missing actual label in order"))?;
        let col = *index
            .get(&label_key(predicted))
            .ok_or_else(|| internal("confusionmat: missing predicted label in order"))?;
        counts[row + col * class_count] += 1.0;
    }
    let matrix = Tensor::new(counts, vec![class_count, class_count])
        .map_err(|err| internal(format!("confusionmat: {err}")))?;
    let mut outputs = vec![Value::Tensor(matrix)];
    if output_limit >= 2 {
        outputs.push(labels_to_value(&order, order_kind)?);
    }
    Ok(outputs)
}

pub(super) fn parse_options(
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<Option<LabelVector>> {
    if rest.is_empty() {
        return Ok(None);
    }
    if !rest.len().is_multiple_of(2) {
        return Err(invalid("confusionmat: name-value options must be paired"));
    }
    let mut order = None;
    let mut index = 0usize;
    while index < rest.len() {
        let name = scalar_text(&rest[index], "option name")?;
        match canonical(&name).as_str() {
            "order" => {
                order = Some(labels_from_value(rest[index + 1].clone(), "Order")?);
            }
            other => {
                return Err(invalid(format!(
                    "confusionmat: unsupported option '{other}'"
                )))
            }
        }
        index += 2;
    }
    if let Some(order) = &order {
        if order.labels.iter().any(is_missing_label) {
            return Err(invalid("confusionmat: Order cannot contain missing labels"));
        }
        if order.labels.is_empty() {
            return Err(invalid(
                "confusionmat: Order must contain at least one label",
            ));
        }
    }
    Ok(order)
}
