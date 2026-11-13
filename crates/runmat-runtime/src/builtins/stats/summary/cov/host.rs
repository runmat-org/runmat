use super::*;

pub(super) async fn cov_host(args: CovArgs) -> BuiltinResult<Value> {
    let CovArgs {
        first,
        second,
        normalization,
        rows,
        weight_vector,
        ..
    } = args;

    let left = value_to_tensor_gather(first).await?;
    let right = match second {
        Some(value) => Some(value_to_tensor_gather(value).await?),
        None => None,
    };
    let expected_weight_rows = if right.is_some() || left.rows() == 1 || left.cols() == 1 {
        left.len()
    } else {
        left.rows()
    };

    let weight_spec = if let Some(weight_value) = weight_vector {
        let vector = value_to_weight_vector(weight_value, expected_weight_rows).await?;
        CovWeightSpec::Vector(vector)
    } else {
        CovWeightSpec::Scalar(normalization)
    };

    let tensor = cov_from_tensors(left, right, rows, weight_spec)?;
    Ok(Value::Tensor(tensor))
}

async fn value_to_tensor_gather(value: Value) -> BuiltinResult<Tensor> {
    match value {
        Value::GpuTensor(handle) => gpu_helpers::gather_tensor_async(&handle).await,
        Value::LogicalArray(logical) => {
            tensor::logical_to_tensor(&logical).map_err(cov_internal_error)
        }
        other => tensor::value_into_tensor_for("cov", other).map_err(cov_internal_error),
    }
}

pub(super) async fn value_to_weight_vector(
    value: Value,
    expected_rows: usize,
) -> BuiltinResult<Vec<f64>> {
    let tensor = match value {
        Value::GpuTensor(handle) => gpu_helpers::gather_tensor_async(&handle).await?,
        Value::LogicalArray(logical) => {
            tensor::logical_to_tensor(&logical).map_err(cov_internal_error)?
        }
        other => tensor::value_into_tensor_for("cov", other).map_err(cov_internal_error)?,
    };
    validate_weight_shape(tensor.shape(), expected_rows)?;
    let values = tensor::tensor_into_values_f64(tensor);
    for (idx, weight) in values.iter().enumerate() {
        if !weight.is_finite() || *weight < 0.0 {
            return Err(cov_error_with_detail(
                &COV_ERROR_INVALID_ARGUMENT,
                format!("weights must be non-negative finite values (index {idx})"),
            ));
        }
    }
    Ok(values)
}
