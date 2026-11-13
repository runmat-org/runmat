use super::*;

pub fn parse_method(value: &Value, name: &'static str) -> BuiltinResult<Option<InterpMethod>> {
    let Some(keyword) = keyword_of(value)? else {
        return Ok(None);
    };
    let method = match keyword.as_str() {
        "linear" => InterpMethod::Linear,
        "nearest" => InterpMethod::Nearest,
        "spline" => InterpMethod::Spline,
        "pchip" => InterpMethod::Pchip,
        "cubic" => InterpMethod::Cubic,
        "makima" => InterpMethod::Makima,
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
    if let Some(keyword) = keyword_of(value)? {
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
    let shape = canonical_shape(tensor.shape(), tensor_len(&tensor));
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
    if !is_vector_shape(tensor.shape()) && tensor_len(&tensor) > 1 {
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

enum SamplePermutation {
    Reverse,
    Sorted(Vec<usize>),
}

enum SampleOrder {
    StrictAscending,
    ValidatedInterp1(Option<SamplePermutation>),
}

/// Validated ordinary-interp1 X plus the exact paired Y reordering recipe.
/// The plan is produced before resident Y is materialized or a provider hook
/// is entered, and consumed once after Y has its canonical series-major form.
pub struct Interp1SamplePlan {
    x: Vec<f64>,
    permutation: Option<SamplePermutation>,
}

impl Interp1SamplePlan {
    pub fn is_ascending(&self) -> bool {
        self.permutation.is_none()
    }

    pub fn x(&self) -> &[f64] {
        &self.x
    }
}

pub async fn prepare_interp1_samples(
    x_value: &Value,
    name: &'static str,
) -> BuiltinResult<Interp1SamplePlan> {
    let x = vector_from_value(x_value.clone(), "X", name).await?;
    let permutation = interp1_sample_permutation(&x, name)?;
    Ok(Interp1SamplePlan { x, permutation })
}

pub async fn interp1_series_from_plan(
    plan: Interp1SamplePlan,
    y_value: Value,
    name: &'static str,
) -> BuiltinResult<NumericSeries> {
    let y_host = materialize_if_resident(y_value).await?;
    let y_tensor = tensor::value_into_tensor_for(name, y_host)
        .map_err(|err| interp_error(name, format!("{name}: {err}")))?;
    series_from_tensor(
        plan.x,
        y_tensor,
        SampleAxis::First,
        SampleOrder::ValidatedInterp1(plan.permutation),
        name,
    )
}

pub async fn series_from_values_on_axis(
    x_value: Value,
    y_value: Value,
    axis: SampleAxis,
    name: &'static str,
) -> BuiltinResult<NumericSeries> {
    let x = vector_from_value(x_value, "X", name).await?;
    validate_breaks(&x, name)?;
    let y_host = materialize_if_resident(y_value).await?;
    let y_tensor = tensor::value_into_tensor_for(name, y_host)
        .map_err(|err| interp_error(name, format!("{name}: {err}")))?;
    series_from_tensor(x, y_tensor, axis, SampleOrder::StrictAscending, name)
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
    let y_shape = canonical_shape(y_tensor.shape(), tensor_len(&y_tensor));
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
        let series = series_from_data(
            x,
            sample_shape,
            values,
            SampleAxis::Last,
            SampleOrder::StrictAscending,
            name,
        )?;
        return Ok(SplineSamples {
            series,
            endpoint_slopes: Some(slopes),
        });
    }
    Ok(SplineSamples {
        series: series_from_data(
            x,
            y_shape,
            y,
            SampleAxis::Last,
            SampleOrder::StrictAscending,
            name,
        )?,
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
    series_from_tensor(
        x,
        y_tensor,
        SampleAxis::First,
        SampleOrder::StrictAscending,
        name,
    )
}

fn series_from_tensor(
    x: Vec<f64>,
    y_tensor: Tensor,
    axis: SampleAxis,
    order: SampleOrder,
    name: &'static str,
) -> BuiltinResult<NumericSeries> {
    let y_shape = canonical_shape(y_tensor.shape(), tensor_len(&y_tensor));
    let y = tensor::tensor_into_values_f64(y_tensor);
    series_from_data(x, y_shape, y, axis, order, name)
}

fn series_from_data(
    x: Vec<f64>,
    y_shape: Vec<usize>,
    y: Vec<f64>,
    axis: SampleAxis,
    order: SampleOrder,
    name: &'static str,
) -> BuiltinResult<NumericSeries> {
    if matches!(&order, SampleOrder::StrictAscending) {
        validate_breaks(&x, name)?;
    }
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

    let mut y = if sample_axis == SampleAxis::Last && series > 1 {
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

    let mut x = x;
    if let SampleOrder::ValidatedInterp1(Some(permutation)) = order {
        permute_interp1_samples(&mut x, &mut y, series, permutation, name)?;
    }
    Ok(NumericSeries {
        x,
        y,
        series,
        component_shape,
        sample_axis,
    })
}

/// The provider's interp1 hook requires ascending host X; nonascending X
/// must be paired with Y on the host before interpolation.
pub fn interp1_x_is_ascending(x: &[f64], name: &'static str) -> BuiltinResult<bool> {
    if x.len() < 2 {
        return Err(interp_error(
            name,
            format!("{name}: X must contain at least two points"),
        ));
    }
    if x.iter().any(|value| !value.is_finite()) {
        return Err(interp_error(name, format!("{name}: X must be finite")));
    }
    Ok(x.windows(2).all(|pair| pair[0] < pair[1]))
}

fn interp1_sample_permutation(
    x: &[f64],
    name: &'static str,
) -> BuiltinResult<Option<SamplePermutation>> {
    if interp1_x_is_ascending(x, name)? {
        return Ok(None);
    }
    let n = x.len();
    if x.windows(2).all(|pair| pair[0] > pair[1]) {
        return Ok(Some(SamplePermutation::Reverse));
    }
    let mut source_at_sorted = Vec::new();
    source_at_sorted.try_reserve_exact(n).map_err(|_| {
        interp_error(
            name,
            format!("{name}: insufficient memory for sample ordering"),
        )
    })?;
    source_at_sorted.extend(0..n);
    source_at_sorted.sort_unstable_by(|&left, &right| x[left].total_cmp(&x[right]));
    if source_at_sorted
        .windows(2)
        .any(|pair| x[pair[0]] == x[pair[1]])
    {
        return Err(interp_error(
            name,
            format!("{name}: X points must be distinct"),
        ));
    }
    Ok(Some(SamplePermutation::Sorted(source_at_sorted)))
}

fn permute_interp1_samples(
    x: &mut [f64],
    y: &mut [f64],
    series: usize,
    permutation: SamplePermutation,
    name: &'static str,
) -> BuiltinResult<()> {
    let n = x.len();
    let SamplePermutation::Sorted(source_at_sorted) = permutation else {
        x.reverse();
        for channel in y.chunks_exact_mut(n).take(series) {
            channel.reverse();
        }
        return Ok(());
    };
    let mut visited = Vec::new();
    visited.try_reserve_exact(n).map_err(|_| {
        interp_error(
            name,
            format!("{name}: insufficient memory for sample ordering"),
        )
    })?;
    visited.resize(n, false);
    permute_samples_in_place(x, &source_at_sorted, &mut visited);
    for channel in y.chunks_exact_mut(n).take(series) {
        permute_samples_in_place(channel, &source_at_sorted, &mut visited);
    }
    Ok(())
}

fn permute_samples_in_place(samples: &mut [f64], source_at_sorted: &[usize], visited: &mut [bool]) {
    visited.fill(false);
    for start in 0..samples.len() {
        if visited[start] {
            continue;
        }
        let first = samples[start];
        let mut current = start;
        loop {
            visited[current] = true;
            let next = source_at_sorted[current];
            if next == start {
                samples[current] = first;
                break;
            }
            samples[current] = samples[next];
            current = next;
        }
    }
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
