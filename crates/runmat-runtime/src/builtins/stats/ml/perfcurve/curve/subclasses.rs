use super::*;

pub(in crate::builtins::stats::ml::perfcurve) fn suby_output(
    observations: &[Observation],
    thresholds: &[ThresholdSpec],
    neg_classes: &[Label],
    xcrit: Criterion,
    ycrit: Criterion,
    cost: &[[f64; 2]; 2],
    xvals: Option<&Vec<f64>>,
    use_nearest: bool,
) -> BuiltinResult<Value> {
    let rows = xvals.map_or(thresholds.len(), Vec::len);
    let cols = neg_classes.len();
    let mut data = suby_buffer(rows, cols)?;
    for col in 0..cols {
        let mut x = Vec::with_capacity(thresholds.len());
        let mut y = Vec::with_capacity(thresholds.len());
        let counts = counts_for_thresholds(observations, thresholds, Some(col));
        for count in &counts {
            x.push(criterion_value(*count, xcrit, cost));
            y.push(criterion_value(*count, ycrit, cost));
        }
        let col_values = if let Some(xvals) = xvals {
            let curve = Curve {
                x,
                y,
                thresholds: thresholds.iter().map(threshold_output_value).collect(),
                counts,
                auc: 0.0,
            };
            xvals
                .iter()
                .map(|target| {
                    if use_nearest {
                        nearest_at_x(&curve, *target).1
                    } else {
                        interpolate_at_x(&curve, *target).1
                    }
                })
                .collect::<Vec<_>>()
        } else {
            y
        };
        for (row, value) in col_values.into_iter().enumerate() {
            data[row + col * rows] = value;
        }
    }
    Ok(Value::Tensor(
        Tensor::new(data, vec![rows, cols]).map_err(|err| internal(format!("perfcurve: {err}")))?,
    ))
}

fn suby_buffer(rows: usize, cols: usize) -> BuiltinResult<Vec<f64>> {
    let len = rows
        .checked_mul(cols)
        .ok_or_else(|| invalid("perfcurve: requested SubY dimensions overflow"))?;
    let mut data = Vec::new();
    data.try_reserve_exact(len)
        .map_err(|_| invalid("perfcurve: requested SubY allocation exceeds available memory"))?;
    data.resize(len, 0.0);
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suby_dimensions_must_fit_before_allocation() {
        let err = suby_buffer(usize::MAX, 2).unwrap_err();
        assert_eq!(err.identifier.as_deref(), ERROR_INVALID_ARGUMENT.identifier);
        assert!(err.message.contains("SubY dimensions overflow"));
    }
}
