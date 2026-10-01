use super::*;

pub fn parse_method(value: &Value, name: &'static str) -> BuiltinResult<Option<InterpMethod>> {
    let Some(keyword) = keyword_of(value) else {
        return Ok(None);
    };
    let method = match keyword.as_str() {
        "linear" => InterpMethod::Linear,
        "nearest" => InterpMethod::Nearest,
        "spline" => InterpMethod::Spline,
        "pchip" | "cubic" => InterpMethod::Pchip,
        _ => {
            return Err(interp_error(
                name,
                format!("{name}: unsupported interpolation method '{keyword}'"),
            ))
        }
    };
    Ok(Some(method))
}

pub async fn parse_extrapolation(
    value: &Value,
    name: &'static str,
) -> BuiltinResult<Option<Extrapolation>> {
    if let Some(keyword) = keyword_of(value) {
        return match keyword.as_str() {
            "extrap" => Ok(Some(Extrapolation::Extrapolate)),
            _ => Ok(None),
        };
    }
    let gathered = dispatcher::gather_if_needed_async(value).await?;
    let Some(scalar) = tensor::scalar_f64_from_value_async(&gathered)
        .await
        .map_err(|err| interp_error(name, format!("{name}: {err}")))?
    else {
        return Ok(None);
    };
    Ok(Some(Extrapolation::Value(scalar)))
}

pub async fn query_points(value: Value, name: &'static str) -> BuiltinResult<QueryPoints> {
    let gathered = materialize_if_resident(value).await?;
    let tensor = tensor::value_into_tensor_for(name, gathered)
        .map_err(|err| interp_error(name, format!("{name}: {err}")))?;
    let shape = canonical_shape(&tensor.shape, tensor_len(&tensor));
    Ok(QueryPoints {
        values: tensor::tensor_into_values_f64(tensor),
        shape,
    })
}

pub async fn vector_from_value(
    value: Value,
    label: &str,
    name: &'static str,
) -> BuiltinResult<Vec<f64>> {
    let gathered = materialize_if_resident(value).await?;
    let tensor = tensor::value_into_tensor_for(name, gathered)
        .map_err(|err| interp_error(name, format!("{name}: {err}")))?;
    if !is_vector_shape(&tensor.shape) && tensor_len(&tensor) > 1 {
        return Err(interp_error(
            name,
            format!("{name}: {label} must be a vector"),
        ));
    }
    Ok(tensor::tensor_into_values_f64(tensor))
}

pub async fn series_from_values(
    x_value: Value,
    y_value: Value,
    name: &'static str,
) -> BuiltinResult<NumericSeries> {
    series_from_values_on_axis(x_value, y_value, SampleAxis::First, name).await
}

pub async fn series_from_values_on_axis(
    x_value: Value,
    y_value: Value,
    axis: SampleAxis,
    name: &'static str,
) -> BuiltinResult<NumericSeries> {
    let x = vector_from_value(x_value, "X", name).await?;
    let y_host = materialize_if_resident(y_value).await?;
    let y_tensor = tensor::value_into_tensor_for(name, y_host)
        .map_err(|err| interp_error(name, format!("{name}: {err}")))?;
    series_from_tensor(x, y_tensor, axis, name)
}

/// Spline interprets two extra values along the final sample axis as the
/// first and last derivatives. PCHIP and interp1 never use this grammar.
pub async fn spline_samples_from_values(
    x_value: Value,
    y_value: Value,
    name: &'static str,
) -> BuiltinResult<SplineSamples> {
    let x = vector_from_value(x_value, "X", name).await?;
    validate_breaks(&x, name)?;
    let y_host = materialize_if_resident(y_value).await?;
    let y_tensor = tensor::value_into_tensor_for(name, y_host)
        .map_err(|err| interp_error(name, format!("{name}: {err}")))?;
    let y_shape = canonical_shape(&y_tensor.shape, tensor_len(&y_tensor));
    let y = tensor::tensor_into_values_f64(y_tensor);
    let endpoint_count = x
        .len()
        .checked_add(2)
        .ok_or_else(|| interp_error(name, format!("{name}: sample count overflow")))?;
    let vector_endpoint_form = y.len() == endpoint_count && is_vector_shape(&y_shape);
    if vector_endpoint_form || y_shape.last().copied() == Some(endpoint_count) {
        let components = y.len() / endpoint_count;
        let mut values = Vec::with_capacity(components * x.len());
        let mut slopes = Vec::with_capacity(components);
        for component in 0..components {
            slopes.push(EndpointSlopes {
                first: y[component],
                last: y[(endpoint_count - 1) * components + component],
            });
        }
        for sample in 0..x.len() {
            values.extend_from_slice(&y[(sample + 1) * components..(sample + 2) * components]);
        }
        let mut sample_shape = if vector_endpoint_form {
            vec![1, x.len()]
        } else {
            y_shape
        };
        *sample_shape.last_mut().expect("canonical tensor shape") = x.len();
        let series = series_from_data(x, sample_shape, values, SampleAxis::Last, name)?;
        return Ok(SplineSamples {
            series,
            endpoint_slopes: Some(slopes),
        });
    }
    Ok(SplineSamples {
        series: series_from_data(x, y_shape, y, SampleAxis::Last, name)?,
        endpoint_slopes: None,
    })
}

pub async fn implicit_series_from_values(
    y_value: Value,
    name: &'static str,
) -> BuiltinResult<NumericSeries> {
    let y_host = materialize_if_resident(y_value).await?;
    let y_tensor = tensor::value_into_tensor_for(name, y_host)
        .map_err(|err| interp_error(name, format!("{name}: {err}")))?;
    let n = first_non_singleton_len(&y_tensor).unwrap_or_else(|| tensor_len(&y_tensor));
    let x: Vec<f64> = (1..=n).map(|value| value as f64).collect();
    series_from_tensor(x, y_tensor, SampleAxis::First, name)
}

fn series_from_tensor(
    x: Vec<f64>,
    y_tensor: Tensor,
    axis: SampleAxis,
    name: &'static str,
) -> BuiltinResult<NumericSeries> {
    let y_shape = canonical_shape(&y_tensor.shape, tensor_len(&y_tensor));
    let y = tensor::tensor_into_values_f64(y_tensor);
    series_from_data(x, y_shape, y, axis, name)
}

fn series_from_data(
    x: Vec<f64>,
    y_shape: Vec<usize>,
    y: Vec<f64>,
    axis: SampleAxis,
    name: &'static str,
) -> BuiltinResult<NumericSeries> {
    validate_breaks(&x, name)?;
    let n = x.len();
    let y_len = y.len();
    if y_len != n && !y_len.is_multiple_of(n) {
        return Err(interp_error(
            name,
            format!("{name}: Y length must match X or contain complete sample series"),
        ));
    }

    let (series, component_shape, sample_axis) = if y_len == n && is_vector_shape(&y_shape) {
        (1, Vec::new(), axis)
    } else if axis == SampleAxis::First && y_shape.first().copied() == Some(n) {
        let series = y_len / n;
        let trailing = if y_shape.len() > 1 {
            y_shape[1..].to_vec()
        } else {
            vec![series]
        };
        (series, trailing, SampleAxis::First)
    } else if axis == SampleAxis::Last && y_shape.last().copied() == Some(n) {
        let series = y_len / n;
        (
            series,
            y_shape[..y_shape.len() - 1].to_vec(),
            SampleAxis::Last,
        )
    } else if axis == SampleAxis::Last && y_shape.first().copied() == Some(n) {
        // Retain RunMat's older first-axis array form when it is unambiguous.
        // The documented last-axis form takes precedence when both match.
        (y_len / n, y_shape[1..].to_vec(), SampleAxis::First)
    } else if y_len == n {
        (1, Vec::new(), axis)
    } else {
        return Err(interp_error(
            name,
            format!("{name}: size of Y must be compatible with X"),
        ));
    };
    if series == 0 {
        return Err(interp_error(
            name,
            format!("{name}: Y must contain sample values"),
        ));
    }

    let y = if sample_axis == SampleAxis::Last && series > 1 {
        let mut series_major = vec![0.0; y.len()];
        for sample in 0..n {
            for component in 0..series {
                series_major[component * n + sample] = y[sample * series + component];
            }
        }
        series_major
    } else {
        y
    };

    Ok(NumericSeries {
        x,
        y,
        series,
        component_shape,
        sample_axis,
    })
}

pub fn validate_breaks(x: &[f64], name: &'static str) -> BuiltinResult<()> {
    if x.len() < 2 {
        return Err(interp_error(
            name,
            format!("{name}: X must contain at least two points"),
        ));
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(interp_error(name, format!("{name}: X must be finite")));
    }
    for pair in x.windows(2) {
        if pair[1] <= pair[0] {
            return Err(interp_error(
                name,
                format!("{name}: X must be strictly increasing"),
            ));
        }
    }
    Ok(())
}
