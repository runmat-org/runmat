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
    let mut data = vec![0.0; rows * cols];
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
