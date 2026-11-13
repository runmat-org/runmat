use super::*;

pub(super) fn cvpartition_compute(
    first: Value,
    second: Value,
    rest: crate::call::FunctionArgumentView<'_>,
    admitted: AdmittedPartition,
    turn: Option<&mut random::RandomTurn>,
) -> BuiltinResult<PartitionSpec> {
    if admitted.custom_first {
        return custom_partition(second);
    }

    let kind = admitted.kind;
    let (value, option_start) = match kind {
        PartitionKind::Leaveout | PartitionKind::Resubstitution => (None, 0usize),
        PartitionKind::KFold | PartitionKind::Holdout => {
            let Some(value) = rest.first() else {
                return Err(invalid_argument(format!(
                    "cvpartition: {} requires a partition value",
                    kind.property_name()
                )));
            };
            (Some(value.clone()), 1usize)
        }
        PartitionKind::Custom => {
            if !rest.is_empty() {
                return Err(invalid_argument(
                    "cvpartition: CustomPartition accepts exactly one test-set argument",
                ));
            }
            return custom_partition(first);
        }
    };
    debug_assert_eq!(option_start, admitted.option_start);
    let mut stratify_option = None;
    for option in admitted.stratify {
        stratify_option = Some(match option {
            AdmittedBoolean::Known(value) => value,
            AdmittedBoolean::Resident(index) => scalar_bool(&rest[index], "Stratify")?,
        });
    }
    let input = PartitionInput::from_value(first)?;
    let stratify = stratify_option.unwrap_or(input.default_stratify());
    if stratify && input.labels.is_none() {
        return Err(invalid_argument(
            "cvpartition: Stratify true requires a grouping variable input",
        ));
    }
    let labels = if stratify {
        input.labels.as_deref()
    } else {
        None
    };
    let n = input.n;
    match kind {
        PartitionKind::KFold => {
            let k = positive_integer(value.as_ref().expect("kfold value"), "KFold")?;
            kfold_partition(
                n,
                k,
                labels,
                turn.expect("random method has an ordered turn"),
            )
        }
        PartitionKind::Holdout => {
            let test_count = holdout_count(value.as_ref().expect("holdout value"), n)?;
            holdout_partition(
                n,
                test_count,
                labels,
                turn.expect("random method has an ordered turn"),
            )
        }
        PartitionKind::Leaveout => leaveout_partition(n),
        PartitionKind::Resubstitution => resubstitution_partition(n),
        PartitionKind::Custom => unreachable!(),
    }
}

pub(super) fn kfold_partition(
    n: usize,
    k: usize,
    labels: Option<&[Option<PartitionLabel>]>,
    turn: &mut random::RandomTurn,
) -> BuiltinResult<PartitionSpec> {
    if n == 0 {
        return Err(invalid_argument("cvpartition: n must be positive"));
    }
    if k < 2 || k >= n {
        return Err(invalid_argument(
            "cvpartition: KFold must be an integer in the interval [2, n)",
        ));
    }
    let len = materialized_mask_len(n, k, "KFold")?;
    let mut test_sets = vec![0u8; len];
    let active_rows = active_rows_for(n, labels);
    if let Some(labels) = labels {
        if labels.len() != n {
            return Err(invalid_argument(
                "cvpartition: stratification labels must match n",
            ));
        }
        for indices in grouped_indices(labels).values() {
            for (offset, row) in shuffled_indices(indices, turn)?.iter().enumerate() {
                test_sets[row + n * (offset % k)] = 1;
            }
        }
    } else {
        for (offset, row) in shuffled_indices(&(0..n).collect::<Vec<_>>(), turn)?
            .iter()
            .enumerate()
        {
            test_sets[row + n * (offset % k)] = 1;
        }
    }
    Ok(PartitionSpec {
        kind: PartitionKind::KFold,
        n,
        test_sets,
        test_set_count: k,
        active_rows,
        is_custom: false,
        is_stratified: labels.is_some(),
    })
}

pub(super) fn holdout_partition(
    n: usize,
    test_count: usize,
    labels: Option<&[Option<PartitionLabel>]>,
    turn: &mut random::RandomTurn,
) -> BuiltinResult<PartitionSpec> {
    if n == 0 {
        return Err(invalid_argument("cvpartition: n must be positive"));
    }
    if test_count == 0 || test_count >= n {
        return Err(invalid_argument(
            "cvpartition: Holdout must select between 1 and n-1 observations",
        ));
    }
    let mut test_sets = vec![0u8; n];
    let active_rows = active_rows_for(n, labels);
    if let Some(labels) = labels {
        let groups = grouped_indices(labels);
        let mut selected = 0usize;
        for indices in groups.values() {
            let group_target = ((indices.len() * test_count) + (n / 2)) / n;
            let group_target = group_target.min(indices.len());
            for row in shuffled_indices(indices, turn)?.iter().take(group_target) {
                if selected < test_count {
                    test_sets[*row] = 1;
                    selected += 1;
                }
            }
        }
        for row in shuffled_indices(&(0..n).collect::<Vec<_>>(), turn)? {
            if selected >= test_count {
                break;
            }
            if active_rows[row] != 0 && test_sets[row] == 0 {
                test_sets[row] = 1;
                selected += 1;
            }
        }
    } else {
        for row in shuffled_indices(&(0..n).collect::<Vec<_>>(), turn)?
            .into_iter()
            .take(test_count)
        {
            test_sets[row] = 1;
        }
    }
    Ok(PartitionSpec {
        kind: PartitionKind::Holdout,
        n,
        test_sets,
        test_set_count: 1,
        active_rows,
        is_custom: false,
        is_stratified: labels.is_some(),
    })
}

pub(super) fn shuffled_indices(
    indices: &[usize],
    turn: &mut random::RandomTurn,
) -> BuiltinResult<Vec<usize>> {
    let mut output = indices.to_vec();
    if output.len() <= 1 {
        return Ok(output);
    }
    let uniforms = random::generate_uniform(turn, output.len() - 1, CVPARTITION_NAME)?;
    for (index, uniform) in uniforms.into_iter().enumerate() {
        let span = output.len() - index;
        let offset = ((uniform * span as f64).floor() as usize).min(span - 1);
        output.swap(index, index + offset);
    }
    Ok(output)
}

pub(super) fn leaveout_partition(n: usize) -> BuiltinResult<PartitionSpec> {
    if n == 0 {
        return Err(invalid_argument("cvpartition: n must be positive"));
    }
    let len = materialized_mask_len(n, n, "Leaveout")?;
    let mut test_sets = vec![0u8; len];
    for row in 0..n {
        test_sets[row + n * row] = 1;
    }
    Ok(PartitionSpec {
        kind: PartitionKind::Leaveout,
        n,
        test_sets,
        test_set_count: n,
        active_rows: vec![1; n],
        is_custom: false,
        is_stratified: false,
    })
}

pub(super) fn resubstitution_partition(n: usize) -> BuiltinResult<PartitionSpec> {
    if n == 0 {
        return Err(invalid_argument("cvpartition: n must be positive"));
    }
    Ok(PartitionSpec {
        kind: PartitionKind::Resubstitution,
        n,
        test_sets: vec![1u8; n],
        test_set_count: 1,
        active_rows: vec![1; n],
        is_custom: false,
        is_stratified: false,
    })
}

pub(super) fn custom_partition(value: Value) -> BuiltinResult<PartitionSpec> {
    let array = match value {
        Value::LogicalArray(array) => array,
        Value::Bool(flag) => LogicalArray::new(vec![u8::from(flag)], vec![1, 1])
            .map_err(|err| internal_error(format!("cvpartition: {err}")))?,
        Value::Tensor(tensor) => {
            return custom_numeric_partition(tensor);
        }
        other => {
            return Err(invalid_argument(format!(
                "cvpartition: CustomPartition test sets must be logical, got {other:?}"
            )))
        }
    };
    let shape = logical_shape(&array)?;
    let n = shape.0;
    let test_set_count = shape.1;
    if n == 0 || test_set_count == 0 {
        return Err(invalid_argument(
            "cvpartition: CustomPartition test sets must be nonempty",
        ));
    }
    Ok(PartitionSpec {
        kind: PartitionKind::Custom,
        n,
        test_sets: array.data.into_vec(),
        test_set_count,
        active_rows: vec![1; n],
        is_custom: true,
        is_stratified: false,
    })
}

pub(super) fn custom_numeric_partition(tensor: Tensor) -> BuiltinResult<PartitionSpec> {
    if tensor.integer_storage().is_some() {
        return custom_integer_partition(tensor);
    }
    let values = tensor::tensor_values_f64_cow(&tensor);
    if values.is_empty() {
        return Err(invalid_argument(
            "cvpartition: CustomPartition test sets must be nonempty",
        ));
    }
    let is_vector = tensor.shape().iter().filter(|dim| **dim > 1).count() <= 1;
    if !is_vector {
        let data = values
            .iter()
            .map(|value| {
                if (*value - 0.0).abs() <= EPS {
                    Ok(0)
                } else if (*value - 1.0).abs() <= EPS {
                    Ok(1)
                } else {
                    Err(invalid_argument(
                        "cvpartition: numeric custom matrices must contain only 0 or 1",
                    ))
                }
            })
            .collect::<BuiltinResult<Vec<_>>>()?;
        let array = LogicalArray::new(data, tensor.shape().to_vec())
            .map_err(|err| internal_error(format!("cvpartition: {err}")))?;
        return custom_partition(Value::LogicalArray(array));
    }

    if values.iter().any(|value| (*value - 0.0).abs() <= EPS) {
        let data = values
            .iter()
            .map(|value| {
                if (*value - 0.0).abs() <= EPS {
                    Ok(0)
                } else if (*value - 1.0).abs() <= EPS {
                    Ok(1)
                } else {
                    Err(invalid_argument(
                        "cvpartition: numeric custom logical vectors must contain only 0 or 1",
                    ))
                }
            })
            .collect::<BuiltinResult<Vec<_>>>()?;
        let array = LogicalArray::new(data, vec![values.len(), 1])
            .map_err(|err| internal_error(format!("cvpartition: {err}")))?;
        return custom_partition(Value::LogicalArray(array));
    }

    let mut ids = Vec::with_capacity(values.len());
    let mut max_id = 0usize;
    for value in values.iter() {
        if !value.is_finite()
            || value.fract() != 0.0
            || *value < 1.0
            || *value > usize::MAX as f64
            || (usize::BITS == 64 && *value == usize::MAX as f64)
        {
            return Err(invalid_argument(
                "cvpartition: custom partition ids must be positive integers",
            ));
        }
        let id = *value as usize;
        max_id = max_id.max(id);
        ids.push(id);
    }
    let n = ids.len();
    let len = materialized_mask_len(n, max_id, "CustomPartition")?;
    let mut test_sets = vec![0u8; len];
    let mut nonempty = vec![0usize; max_id];
    for (row, id) in ids.into_iter().enumerate() {
        test_sets[row + n * (id - 1)] = 1;
        nonempty[id - 1] += 1;
    }
    if nonempty.contains(&0) {
        return Err(invalid_argument(
            "cvpartition: custom partition ids must form nonempty test sets",
        ));
    }
    Ok(PartitionSpec {
        kind: PartitionKind::Custom,
        n,
        test_sets,
        test_set_count: max_id,
        active_rows: vec![1; n],
        is_custom: true,
        is_stratified: false,
    })
}

pub(super) fn custom_integer_partition(tensor: Tensor) -> BuiltinResult<PartitionSpec> {
    let storage = tensor
        .integer_storage()
        .expect("custom_integer_partition requires integer storage");
    let values = storage.exact_values();
    if values.is_empty() {
        return Err(invalid_argument(
            "cvpartition: CustomPartition test sets must be nonempty",
        ));
    }
    let is_vector = tensor.shape().iter().filter(|dim| **dim > 1).count() <= 1;
    let logical_byte = |value: &runmat_value::IntValue| match value.try_to_usize() {
        Some(0) => Ok(0),
        Some(1) => Ok(1),
        _ => Err(invalid_argument(
            "cvpartition: numeric custom logical values must contain only 0 or 1",
        )),
    };
    if !is_vector || values.iter().any(|value| value.is_zero()) {
        let data = values
            .iter()
            .map(logical_byte)
            .collect::<BuiltinResult<Vec<_>>>()?;
        let shape = if is_vector {
            vec![values.len(), 1]
        } else {
            tensor.shape().to_vec()
        };
        let array = LogicalArray::new(data, shape)
            .map_err(|err| internal_error(format!("cvpartition: {err}")))?;
        return custom_partition(Value::LogicalArray(array));
    }

    let mut ids = Vec::with_capacity(values.len());
    let mut max_id = 0usize;
    for value in values {
        let id = value.try_to_usize().filter(|id| *id > 0).ok_or_else(|| {
            invalid_argument("cvpartition: custom partition ids must be positive integers")
        })?;
        max_id = max_id.max(id);
        ids.push(id);
    }
    custom_partition_from_ids(ids, max_id)
}

pub(super) fn custom_partition_from_ids(
    ids: Vec<usize>,
    max_id: usize,
) -> BuiltinResult<PartitionSpec> {
    let n = ids.len();
    let len = materialized_mask_len(n, max_id, "CustomPartition")?;
    let mut test_sets = vec![0u8; len];
    let mut nonempty = vec![0usize; max_id];
    for (row, id) in ids.into_iter().enumerate() {
        test_sets[row + n * (id - 1)] = 1;
        nonempty[id - 1] += 1;
    }
    if nonempty.contains(&0) {
        return Err(invalid_argument(
            "cvpartition: custom partition ids must form nonempty test sets",
        ));
    }
    Ok(PartitionSpec {
        kind: PartitionKind::Custom,
        n,
        test_sets,
        test_set_count: max_id,
        active_rows: vec![1; n],
        is_custom: true,
        is_stratified: false,
    })
}

pub(super) fn grouped_indices(
    labels: &[Option<PartitionLabel>],
) -> BTreeMap<PartitionLabel, Vec<usize>> {
    let mut groups = BTreeMap::<PartitionLabel, Vec<usize>>::new();
    for (idx, label) in labels.iter().enumerate() {
        if let Some(label) = label {
            groups.entry(label.clone()).or_default().push(idx);
        }
    }
    groups
}

pub(super) fn active_rows_for(n: usize, labels: Option<&[Option<PartitionLabel>]>) -> Vec<u8> {
    match labels {
        Some(labels) => labels
            .iter()
            .map(|label| u8::from(label.is_some()))
            .collect(),
        None => vec![1; n],
    }
}
