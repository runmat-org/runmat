use super::*;

#[derive(Clone, Copy)]
struct VisibleShape {
    rows: usize,
    columns: usize,
    rank: usize,
    count: usize,
}

fn visible_shape(value: &Value) -> BuiltinResult<VisibleShape> {
    let shape: &[usize] = match value {
        Value::Tensor(value) => value.shape(),
        Value::ComplexTensor(value) => value.shape(),
        Value::LogicalArray(value) => &value.shape,
        Value::GpuTensor(handle) => handle.shape(),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => &[1, 1],
        _ => return Err(invalid("lscov: expected dense numeric or logical input")),
    };
    let count = shape
        .iter()
        .try_fold(1usize, |count, &extent| count.checked_mul(extent))
        .ok_or_else(|| invalid("lscov: input dimensions overflow"))?;
    // Dense one-dimensional values are row vectors in the canonical tensor
    // representation; use the same geometry before any resident download.
    let (rows, columns) = match shape {
        [columns] => (1, *columns),
        [rows, columns, ..] => (*rows, *columns),
        [] => (0, 0),
    };
    Ok(VisibleShape {
        rows,
        columns,
        rank: shape.len(),
        count,
    })
}

pub(super) fn preflight_geometry(
    a: &Value,
    b: &Value,
    header: &mut LscovArguments,
    outputs: Option<usize>,
) -> BuiltinResult<()> {
    let a = visible_shape(a)?;
    let b = visible_shape(b)?;
    if a.rank > 2 || a.rows == 0 || a.columns == 0 {
        return Err(invalid("lscov: A must be a nonempty 2-D matrix"));
    }
    if b.rank > 2 {
        return Err(invalid("lscov: B must be a 2-D matrix"));
    }
    let b_vector = b.rows == 1 || b.columns == 1;
    let response_columns = if b_vector && b.count == a.rows {
        1
    } else if b.rows == a.rows {
        b.columns
    } else {
        return Err(invalid(
            "lscov: B must have one row per observation or be an observation vector",
        ));
    };
    if outputs == Some(4) && response_columns != 1 {
        return Err(invalid(
            "lscov: fourth output S is supported only when B is a vector",
        ));
    }
    if outputs != Some(0) {
        ensure_budget(a.rows, a.columns, "design matrix")?;
        ensure_work_budget(a.columns)?;
        ensure_output_budget(a.columns, response_columns, outputs.unwrap_or(1))?;
    }
    if let Some(WeightingInput::Pending(weight)) = &header.weighting {
        if !matches!(
            weight,
            Value::Tensor(_)
                | Value::LogicalArray(_)
                | Value::GpuTensor(_)
                | Value::Num(_)
                | Value::Int(_)
                | Value::Bool(_)
        ) {
            return Err(invalid("lscov: V must be real numeric or logical"));
        }
        if matches!(weight, Value::GpuTensor(handle) if handle.physical().storage() != runmat_accelerate_api::GpuTensorStorage::Real)
        {
            return Err(invalid("lscov: V must be real numeric or logical"));
        }
        let weight = visible_shape(weight)?;
        if weight.count > 0 {
            let vector = weight.rank <= 2 && (weight.rows == 1 || weight.columns == 1);
            if vector && weight.count != 1 && weight.count != a.rows {
                return Err(invalid(
                    "lscov: V vector length must match the number of rows in A",
                ));
            }
            if !vector && (weight.rank > 2 || weight.rows != a.rows || weight.columns != a.rows) {
                return Err(invalid("lscov: V must be empty, a weight vector, or a square covariance matrix matching A rows"));
            }
            if !vector {
                ensure_budget(a.rows, a.rows, "covariance matrix")?;
            }
        }
    }
    if !matches!(
        header.weighting.as_ref(),
        Some(WeightingInput::Pending(Value::GpuTensor(_)))
    ) {
        if let Some(WeightingInput::Pending(weight)) = header.weighting.take() {
            header.weighting = Some(WeightingInput::Prepared(parse_weighting_host(
                weight, a.rows,
            )?));
        }
    }
    Ok(())
}

pub(super) async fn gather_numeric_matrix(value: Value) -> BuiltinResult<NumericMatrix> {
    let gathered = gather_if_needed_async(&value).await?;
    match gathered {
        Value::ComplexTensor(tensor) => Ok(NumericMatrix::Complex(tensor)),
        Value::Complex(re, im) => ComplexTensor::new(vec![(re, im)], vec![1, 1])
            .map(NumericMatrix::Complex)
            .map_err(|err| invalid(format!("lscov: {err}"))),
        other => {
            let tensor = tensor::value_into_tensor_for(NAME, other)
                .map_err(|err| invalid(format!("lscov: {err}")))?;
            Ok(NumericMatrix::Real(tensor))
        }
    }
}

pub(super) struct LscovArguments {
    weighting: Option<WeightingInput>,
    algorithm: Algorithm,
}

pub(super) enum WeightingInput {
    Prepared(Weighting),
    Pending(Value),
}

pub(super) fn parse_header(rest: crate::call::BuiltinArguments) -> BuiltinResult<LscovArguments> {
    let mut arguments = rest.into_iter();
    let first = arguments.next();
    let second = arguments.next();
    let (weighting, algorithm) = match (first, second) {
        (None, None) => (None, Algorithm::Chol),
        (Some(value), None) => match keyword_of(&value)? {
            Some(keyword) => (None, parse_algorithm(&keyword)?),
            None => (Some(value), Algorithm::Chol),
        },
        (Some(weighting), Some(alg)) => {
            let keyword = keyword_of(&alg)?.ok_or_else(|| {
                invalid("lscov: algorithm must be a string scalar or character row")
            })?;
            (Some(weighting), parse_algorithm(&keyword)?)
        }
        (None, Some(_)) => unreachable!("second argument requires a first"),
    };
    Ok(LscovArguments {
        weighting: weighting.map(WeightingInput::Pending),
        algorithm,
    })
}

pub(super) async fn finish_header(
    header: LscovArguments,
    rows: usize,
) -> BuiltinResult<ParsedArgs> {
    let weighting = match header.weighting {
        Some(WeightingInput::Prepared(value)) => value,
        Some(WeightingInput::Pending(value)) => parse_weighting(value, rows).await?,
        None => Weighting::Identity,
    };
    Ok(ParsedArgs {
        weighting,
        algorithm: header.algorithm,
    })
}

pub(super) fn parse_algorithm(keyword: &str) -> BuiltinResult<Algorithm> {
    Algorithm::from_public_keyword(keyword)
        .ok_or_else(|| invalid("lscov: algorithm must be \"orth\" or \"chol\""))
}

pub(super) async fn parse_weighting(value: Value, rows: usize) -> BuiltinResult<Weighting> {
    let gathered = gather_if_needed_async(&value).await?;
    parse_weighting_host(gathered, rows)
}

fn parse_weighting_host(value: Value, rows: usize) -> BuiltinResult<Weighting> {
    let tensor = tensor::value_into_tensor_for(NAME, value)
        .map_err(|err| invalid(format!("lscov: {err}")))?;
    if tensor::tensor_element_len(&tensor) == 0 {
        return Ok(Weighting::Identity);
    }
    let values = tensor::tensor_values_f64(&tensor);
    if is_vector(&tensor) {
        let weights = if values.len() == 1 && rows > 1 {
            vec![values[0]; rows]
        } else {
            values
        };
        if weights.len() != rows {
            return Err(invalid(
                "lscov: V vector length must match the number of rows in A",
            ));
        }
        if !weights.iter().any(|weight| *weight > 0.0) {
            return Err(invalid(
                "lscov: V weights must include at least one positive value",
            ));
        }
        for weight in &weights {
            if !weight.is_finite() || *weight < 0.0 {
                return Err(invalid(
                    "lscov: V weights must be finite nonnegative values",
                ));
            }
        }
        return Ok(Weighting::Weights(weights));
    }
    if tensor.shape().len() > 2 || lscov_dense_dimensions(&tensor) != (rows, rows) {
        return Err(invalid(
            "lscov: V must be empty, a weight vector, or a square covariance matrix matching A rows",
        ));
    }
    ensure_budget(rows, rows, "covariance matrix")?;
    let cov = DMatrix::from_column_slice(rows, rows, &values);
    validate_symmetric_covariance(&cov)?;
    Ok(Weighting::Covariance(cov))
}
