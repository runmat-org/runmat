//! Exact integer bin membership, floating projection and chart counts.
use super::input::ensure_compatible_lengths;
use super::plot::build_surface_from_counts;
use super::*;
use std::cmp::Ordering;

struct AxisProjection {
    values: Vec<f64>,
    limits: Option<(f64, f64)>,
    output_origin: f64,
    exact_values: Option<Vec<i128>>,
    exact_bin_limits: Option<(i128, i128)>,
}

fn project_axis(
    data: &Tensor,
    explicit: Option<&Tensor>,
    name: &str,
) -> BuiltinResult<AxisProjection> {
    if data.integer_storage().is_some()
        && explicit.is_none_or(|limits| limits.integer_storage().is_some())
    {
        let values = (0..data.len())
            .map(|index| {
                data.numeric_value_at(index)
                    .and_then(integer_i128)
                    .expect("integer tensor values remain exact")
            })
            .collect::<Vec<_>>();
        let exact_limits = explicit.map(|limits| {
            (
                integer_i128(
                    limits
                        .numeric_value_at(0)
                        .expect("validated two-element limits"),
                )
                .expect("integer limits remain exact"),
                integer_i128(
                    limits
                        .numeric_value_at(1)
                        .expect("validated two-element limits"),
                )
                .expect("integer limits remain exact"),
            )
        });
        let origin = exact_limits
            .map(|limits| limits.0)
            .or_else(|| values.iter().copied().min())
            .unwrap_or(0);
        let projected_limits =
            exact_limits.map(|(lo, hi)| ((lo - origin) as f64, (hi - origin) as f64));
        let exact_bin_limits = exact_limits.or_else(|| {
            let lo = values.iter().copied().min()?;
            let hi = values.iter().copied().max()?;
            (lo < hi).then_some((lo, hi))
        });
        return Ok(AxisProjection {
            values: values
                .iter()
                .copied()
                .map(|value| (value - origin) as f64)
                .collect(),
            limits: projected_limits,
            output_origin: origin as f64,
            exact_values: Some(values),
            exact_bin_limits,
        });
    }

    let limits = explicit
        .map(|limits| {
            validate_limits(
                limits
                    .numeric_value_at(0)
                    .expect("validated two-element limits")
                    .materialize_f64(),
                limits
                    .numeric_value_at(1)
                    .expect("validated two-element limits")
                    .materialize_f64(),
                name,
            )
        })
        .transpose()?;
    Ok(AxisProjection {
        values: tensor_values_f64(data),
        limits,
        output_origin: 0.0,
        exact_values: None,
        exact_bin_limits: None,
    })
}

pub(crate) fn validate_face_alpha(value: f64) -> BuiltinResult<()> {
    if !(0.0..=1.0).contains(&value) || !value.is_finite() {
        return Err(invalid(
            "binscatter: FaceAlpha must be in the interval [0,1]",
        ));
    }
    Ok(())
}

pub(crate) fn build_binscatter_chart(
    x_data: &Tensor,
    y_data: &Tensor,
    num_bins: [usize; 2],
    x_limits: Option<&Tensor>,
    y_limits: Option<&Tensor>,
    show_empty_bins: bool,
    face_alpha: f64,
    display_name: Option<&str>,
    color_limits: Option<(f64, f64)>,
) -> BuiltinResult<BinscatterChart> {
    ensure_compatible_lengths(x_data, y_data)?;
    validate_bin_counts(num_bins)?;
    validate_face_alpha(face_alpha)?;

    let x_projection = project_axis(x_data, x_limits, "XLimits")?;
    let y_projection = project_axis(y_data, y_limits, "YLimits")?;
    let x_limits = derive_axis_limits(&x_projection.values, x_projection.limits, "XLimits")?;
    let y_limits = derive_axis_limits(&y_projection.values, y_projection.limits, "YLimits")?;
    let x_edges = linspace(x_limits.0, x_limits.1, num_bins[0] + 1);
    let y_edges = linspace(y_limits.0, y_limits.1, num_bins[1] + 1);
    validate_edges(
        &x_edges,
        "XBinEdges",
        x_projection.exact_bin_limits.is_none(),
    )?;
    validate_edges(
        &y_edges,
        "YBinEdges",
        y_projection.exact_bin_limits.is_none(),
    )?;
    let cell_count = num_bins[0]
        .checked_mul(num_bins[1])
        .ok_or_else(|| invalid("binscatter: NumBins product is too large"))?;
    let mut counts = vec![0.0; cell_count];
    for (index, (&x, &y)) in x_projection
        .values
        .iter()
        .zip(y_projection.values.iter())
        .enumerate()
    {
        if !x.is_finite() || !y.is_finite() {
            continue;
        }
        let ix = if x_projection.exact_bin_limits.is_some() {
            exact_projected_bin_index(&x_projection, index, num_bins[0])
        } else {
            find_bin_index(x, &x_edges)
        };
        let iy = if y_projection.exact_bin_limits.is_some() {
            exact_projected_bin_index(&y_projection, index, num_bins[1])
        } else {
            find_bin_index(y, &y_edges)
        };
        if let (Some(ix), Some(iy)) = (ix, iy) {
            counts[ix + iy * num_bins[0]] += 1.0;
        }
    }
    let values = Tensor::new(counts.clone(), vec![num_bins[0], num_bins[1]])
        .map_err(|err| internal(format!("binscatter: {err}")))?;
    let output_x_edges = x_edges
        .iter()
        .map(|edge| edge + x_projection.output_origin)
        .collect::<Vec<_>>();
    let output_y_edges = y_edges
        .iter()
        .map(|edge| edge + y_projection.output_origin)
        .collect::<Vec<_>>();
    validate_edges(
        &output_x_edges,
        "XBinEdges",
        x_projection.exact_bin_limits.is_none(),
    )?;
    validate_edges(
        &output_y_edges,
        "YBinEdges",
        y_projection.exact_bin_limits.is_none(),
    )?;
    let surface = build_surface_from_counts(
        &counts,
        &output_x_edges,
        &output_y_edges,
        num_bins,
        show_empty_bins,
        face_alpha,
        display_name,
        color_limits,
    )?;
    Ok(BinscatterChart {
        values,
        x_bin_edges: output_x_edges,
        y_bin_edges: output_y_edges,
        surface,
    })
}

fn exact_projected_bin_index(
    projection: &AxisProjection,
    index: usize,
    bins: usize,
) -> Option<usize> {
    let value = *projection.exact_values.as_ref()?.get(index)?;
    let (lo, hi) = projection.exact_bin_limits?;
    if value < lo || value > hi {
        return None;
    }
    if value == hi {
        return Some(bins - 1);
    }
    let offset = value - lo;
    let span = hi - lo;
    usize::try_from(offset * bins as i128 / span).ok()
}

pub(crate) fn auto_bin_count(
    data: &Tensor,
    limits: Option<&Tensor>,
    cap: usize,
) -> BuiltinResult<usize> {
    let projection = project_axis(data, limits, "Limits")?;
    let limits = derive_axis_limits(&projection.values, projection.limits, "Limits")?;
    let filtered = projection
        .values
        .iter()
        .copied()
        .filter(|value| value.is_finite() && *value >= limits.0 && *value <= limits.1)
        .collect::<Vec<_>>();
    if filtered.len() < 2 {
        return Ok(1);
    }
    let width = scott_width(&filtered);
    if !width.is_finite() || width <= 0.0 {
        return Ok(1);
    }
    let bins = ((limits.1 - limits.0) / width).ceil().max(1.0) as usize;
    Ok(bins.clamp(1, cap.min(MAX_NUM_BINS)))
}

fn validate_bin_counts(num_bins: [usize; 2]) -> BuiltinResult<()> {
    for count in num_bins {
        if count == 0 || count > MAX_NUM_BINS {
            return Err(invalid(format!(
                "binscatter: NumBins entries must be in the range 1..={MAX_NUM_BINS}"
            )));
        }
    }
    Ok(())
}

fn derive_axis_limits(
    data: &[f64],
    explicit: Option<(f64, f64)>,
    name: &str,
) -> BuiltinResult<(f64, f64)> {
    if let Some((lo, hi)) = explicit {
        return validate_limits(lo, hi, name);
    }
    let mut min_value: Option<f64> = None;
    let mut max_value: Option<f64> = None;
    for value in data.iter().copied().filter(|value| value.is_finite()) {
        min_value = Some(min_value.map_or(value, |current| current.min(value)));
        max_value = Some(max_value.map_or(value, |current| current.max(value)));
    }
    let (mut lo, mut hi) = match (min_value, max_value) {
        (Some(lo), Some(hi)) => (lo, hi),
        _ => (0.0, 1.0),
    };
    if lo == hi {
        let pad = lo.abs().max(1.0) * 0.5;
        if (hi + pad).is_finite() {
            hi += pad;
        }
        if (lo - pad).is_finite() {
            lo -= pad;
        }
        if lo == hi {
            return Err(invalid(format!(
                "binscatter: {name} cannot be represented by finite bin edges"
            )));
        }
    }
    Ok((lo, hi))
}

fn scott_width(values: &[f64]) -> f64 {
    let factor = 3.5 / (values.len() as f64).powf(1.0 / 3.0);
    super::super::dispersion::finite_sample_std_scaled(values, factor).unwrap_or(0.0)
}

fn linspace(start: f64, stop: f64, count: usize) -> Vec<f64> {
    if count <= 1 {
        return vec![start];
    }
    let mut out = Vec::with_capacity(count);
    for idx in 0..count {
        let fraction = idx as f64 / (count - 1) as f64;
        out.push(start.mul_add(1.0 - fraction, stop * fraction));
    }
    if let Some(last) = out.last_mut() {
        *last = stop;
    }
    out
}

fn validate_edges(edges: &[f64], name: &str, require_distinct: bool) -> BuiltinResult<()> {
    if edges.iter().any(|edge| !edge.is_finite())
        || edges
            .windows(2)
            .any(|pair| pair[0] > pair[1] || (require_distinct && pair[0] == pair[1]))
    {
        return Err(invalid(format!(
            "binscatter: {name} cannot represent distinct finite bin boundaries"
        )));
    }
    Ok(())
}

fn find_bin_index(value: f64, edges: &[f64]) -> Option<usize> {
    if value < edges[0] || value > edges[edges.len() - 1] {
        return None;
    }
    if value == edges[edges.len() - 1] {
        return Some(edges.len() - 2);
    }
    match edges.binary_search_by(|edge| edge.partial_cmp(&value).unwrap_or(Ordering::Less)) {
        Ok(index) => {
            if index == 0 {
                Some(0)
            } else if index < edges.len() - 1 {
                Some(index)
            } else {
                Some(edges.len() - 2)
            }
        }
        Err(index) => {
            if index == 0 || index > edges.len() - 1 {
                None
            } else {
                Some(index - 1)
            }
        }
    }
}

#[cfg(test)]
mod finite_edge_tests {
    use super::*;

    #[test]
    fn automatic_width_reuses_scaled_finite_dispersion() {
        for scale in [1e200, 1e-200] {
            let width = scott_width(&[scale, 2.0 * scale]);
            assert!(width.is_finite() && width > 0.0, "{width}");
        }
        assert_eq!(scott_width(&[1.0]), 0.0);
        assert_eq!(scott_width(&[2.0, 2.0]), 0.0);
    }

    #[test]
    fn opposite_finite_extremes_have_finite_monotone_edges() {
        let edges = linspace(-f64::MAX, f64::MAX, 5);
        validate_edges(&edges, "XBinEdges", true).unwrap();
        assert_eq!(edges.first().copied(), Some(-f64::MAX));
        assert_eq!(edges[2], 0.0);
        assert_eq!(edges.last().copied(), Some(f64::MAX));
    }

    #[test]
    fn constant_realmax_has_one_sided_finite_padding() {
        let limits = derive_axis_limits(&[f64::MAX, f64::MAX], None, "XLimits").unwrap();
        assert!(limits.0.is_finite() && limits.0 < limits.1);
        assert_eq!(limits.1, f64::MAX);
        let edges = linspace(limits.0, limits.1, 3);
        validate_edges(&edges, "XBinEdges", true).unwrap();
    }

    #[test]
    fn extreme_observations_publish_finite_chart_edges() {
        let data = Tensor::new(vec![-f64::MAX, f64::MAX], vec![2, 1]).unwrap();
        let chart =
            build_binscatter_chart(&data, &data, [2, 2], None, None, false, 1.0, None, None)
                .unwrap();
        validate_edges(&chart.x_bin_edges, "XBinEdges", true).unwrap();
        validate_edges(&chart.y_bin_edges, "YBinEdges", true).unwrap();
        assert_eq!(chart.x_bin_edges, vec![-f64::MAX, 0.0, f64::MAX]);
        assert_eq!(chart.values.materialize_f64(), vec![1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn unrepresentable_floating_bin_boundary_rejects_before_publication() {
        let adjacent = f64::from_bits(1.0f64.to_bits() + 1);
        let x = Tensor::new(vec![1.0, adjacent], vec![2, 1]).unwrap();
        let y = Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap();
        let error = build_binscatter_chart(&x, &y, [2, 2], None, None, false, 1.0, None, None)
            .err()
            .expect("a binary64 midpoint does not exist between adjacent numbers");
        assert!(error.message().contains("distinct finite bin boundaries"));
    }
}
