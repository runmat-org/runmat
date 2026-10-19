use super::*;

pub(super) fn evaluate_logical(
    kind: TopKKind,
    logical: LogicalArray,
    args: &TopKArgs,
) -> BuiltinResult<TopKEvaluation> {
    let shape = logical.shape.clone();
    let tensor = Tensor::new_integer(IntegerStorage::U8(logical.data.into_vec()), shape)
        .map_err(|message| topk_internal(kind, message))?;
    let evaluation = evaluate_real(kind, tensor, args)?;
    let values = match evaluation.values {
        Value::Int(IntValue::U8(value)) => Value::Bool(value != 0),
        Value::Tensor(tensor) => {
            let shape = tensor.shape().to_vec();
            let IntegerStorage::U8(values) = tensor
                .into_numeric_storage()
                .map_err(|error| topk_internal(kind, error))?
                .into_integer_storage()
                .map_err(|_| topk_internal(kind, "logical selection lost integer storage"))?
            else {
                return Err(topk_internal(
                    kind,
                    "logical selection changed its storage class",
                ));
            };
            Value::LogicalArray(
                LogicalArray::new(values, shape).map_err(|message| topk_internal(kind, message))?,
            )
        }
        other => {
            return Err(topk_internal(
                kind,
                format!("logical selection produced unexpected value {other:?}"),
            ))
        }
    };
    Ok(TopKEvaluation {
        values,
        indices: evaluation.indices,
    })
}

pub(super) fn evaluate_real(
    kind: TopKKind,
    tensor: Tensor,
    args: &TopKArgs,
) -> BuiltinResult<TopKEvaluation> {
    let shape = normalize_shape(tensor.shape().to_vec());
    let dim = selected_dim(&shape, args.dim);
    let axis = dim.saturating_sub(1);
    let axis_len = shape.get(axis).copied().unwrap_or(1);
    let take = args.k.min(axis_len);
    if axis >= shape.len() && take != 0 {
        let indices = Tensor::new(vec![1.0; tensor.len()], shape)
            .map_err(|message| topk_internal(kind, message))?;
        return Ok(TopKEvaluation {
            values: tensor::tensor_into_value(tensor),
            indices: tensor::tensor_into_value(indices),
        });
    }
    let output_shape = output_shape_for_topk(kind, &shape, axis, take)?;
    let storage = tensor
        .into_numeric_storage()
        .map_err(|message| topk_internal(kind, message))?;
    if storage.is_empty() || take == 0 {
        let values = Tensor::from_numeric_storage(storage.zeros_like(0), output_shape.clone())
            .map_err(|message| topk_internal(kind, message))?;
        let indices = Tensor::new(Vec::new(), output_shape)
            .map_err(|message| topk_internal(kind, message))?;
        return Ok(TopKEvaluation {
            values: tensor::tensor_into_value(values),
            indices: tensor::tensor_into_value(indices),
        });
    }

    let input_strides = compute_strides(&shape);
    let output_strides = compute_strides(&output_shape);
    let output_len = checked_element_count(kind, &output_shape)?;
    let mut selected = vec![0usize; output_len];
    let mut indices = vec![0.0; output_len];
    let mut coords = vec![0usize; output_shape.len()];
    for out_base in 0..output_len {
        if coords.get(axis).copied().unwrap_or(0) != 0 {
            increment_coords(&mut coords, &output_shape);
            continue;
        }
        let mut entries = Vec::with_capacity(axis_len);
        for reduce_idx in 0..axis_len {
            let mut input_coords = coords.clone();
            if axis >= input_coords.len() {
                input_coords.resize(axis + 1, 0);
            }
            input_coords[axis] = reduce_idx;
            let input_index = map_linear_index(&input_coords, &input_strides);
            entries.push(RealEntry {
                value: storage.value_at(input_index).ok_or_else(|| {
                    topk_internal(kind, format!("input index {input_index} is out of bounds"))
                })?,
                index: reduce_idx,
                source_index: input_index,
            });
        }
        entries.sort_by(|a, b| compare_real_entries(kind, args.comparison, a, b));
        for (rank, entry) in entries.iter().take(take).enumerate() {
            let mut out_coords = coords.clone();
            out_coords[axis] = rank;
            let out_idx = map_linear_index(&out_coords, &output_strides);
            selected[out_idx] = entry.source_index;
            indices[out_idx] = (entry.index + 1) as f64;
        }
        let _ = out_base;
        increment_coords(&mut coords, &output_shape);
    }

    let values = storage
        .gather(&selected)
        .and_then(|values| Tensor::from_numeric_storage(values, output_shape.clone()))
        .map_err(|message| topk_internal(kind, message))?;
    let indices =
        Tensor::new(indices, output_shape).map_err(|message| topk_internal(kind, message))?;
    Ok(TopKEvaluation {
        values: tensor::tensor_into_value(values),
        indices: tensor::tensor_into_value(indices),
    })
}

pub(super) fn evaluate_complex(
    kind: TopKKind,
    tensor: ComplexTensor,
    args: &TopKArgs,
) -> BuiltinResult<TopKEvaluation> {
    let shape = normalize_shape(tensor.shape().to_vec());
    let storage = tensor.into_complex_storage();
    let comparison_values = storage.materialize_f64();
    let dim = selected_dim(&shape, args.dim);
    let axis = dim.saturating_sub(1);
    let axis_len = shape.get(axis).copied().unwrap_or(1);
    let take = args.k.min(axis_len);
    if axis >= shape.len() && take != 0 {
        let indices = Tensor::new(vec![1.0; storage.len()], shape.clone())
            .map_err(|message| topk_internal(kind, message))?;
        let values = ComplexTensor::from_complex_storage(storage, shape)
            .map_err(|message| topk_internal(kind, message))?;
        return Ok(TopKEvaluation {
            values: complex_tensor_into_value(values),
            indices: tensor::tensor_into_value(indices),
        });
    }
    let output_shape = output_shape_for_topk(kind, &shape, axis, take)?;
    if storage.is_empty() || take == 0 {
        let values = storage
            .gather(&[])
            .and_then(|storage| ComplexTensor::from_complex_storage(storage, output_shape.clone()))
            .map_err(|message| topk_internal(kind, message))?;
        let indices = Tensor::new(Vec::new(), output_shape)
            .map_err(|message| topk_internal(kind, message))?;
        return Ok(TopKEvaluation {
            values: complex_tensor_into_value(values),
            indices: tensor::tensor_into_value(indices),
        });
    }

    let input_strides = compute_strides(&shape);
    let output_strides = compute_strides(&output_shape);
    let output_len = checked_element_count(kind, &output_shape)?;
    let mut selected = vec![0usize; output_len];
    let mut indices = vec![0.0; output_len];
    let mut coords = vec![0usize; output_shape.len()];
    for out_base in 0..output_len {
        if coords.get(axis).copied().unwrap_or(0) != 0 {
            increment_coords(&mut coords, &output_shape);
            continue;
        }
        let mut entries = Vec::with_capacity(axis_len);
        for reduce_idx in 0..axis_len {
            let mut input_coords = coords.clone();
            if axis >= input_coords.len() {
                input_coords.resize(axis + 1, 0);
            }
            input_coords[axis] = reduce_idx;
            let input_index = map_linear_index(&input_coords, &input_strides);
            entries.push(ComplexEntry {
                value: comparison_values[input_index],
                index: reduce_idx,
                source_index: input_index,
            });
        }
        entries.sort_by(|a, b| compare_complex_entries(kind, args.comparison, a, b));
        for (rank, entry) in entries.iter().take(take).enumerate() {
            let mut out_coords = coords.clone();
            out_coords[axis] = rank;
            let out_idx = map_linear_index(&out_coords, &output_strides);
            selected[out_idx] = entry.source_index;
            indices[out_idx] = (entry.index + 1) as f64;
        }
        let _ = out_base;
        increment_coords(&mut coords, &output_shape);
    }

    let values = storage
        .gather(&selected)
        .and_then(|storage| ComplexTensor::from_complex_storage(storage, output_shape.clone()))
        .map_err(|message| topk_internal(kind, message))?;
    let indices =
        Tensor::new(indices, output_shape).map_err(|message| topk_internal(kind, message))?;
    Ok(TopKEvaluation {
        values: complex_tensor_into_value(values),
        indices: tensor::tensor_into_value(indices),
    })
}
