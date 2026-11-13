use super::*;
pub(super) async fn evaluate(
    data: &groups::Groups,
    cooperation: &mut Cooperation<'_>,
) -> BuiltinResult<Vec<Value>> {
    let mut moments = Vec::with_capacity(data.values.len());
    for values in &data.values {
        moments.push(inferential::moments(values, cooperation).await?);
    }
    let n = moments
        .iter()
        .try_fold(0usize, |n, m| n.checked_add(m.count))
        .ok_or_else(|| invalid(NAME, "observation count overflow"))?;
    let k = moments.len();
    if n <= k {
        return Err(invalid(
            NAME,
            "ANOVA requires positive within-group degrees of freedom",
        ));
    }
    let origin = moments[0].mean;
    let mut grand = origin
        + moments
            .iter()
            .map(|m| (m.mean - origin) * (m.count as f64 / n as f64))
            .sum::<f64>();
    if !grand.is_finite() && moments.iter().all(|m| m.mean.is_finite()) {
        let scale = moments.iter().map(|m| m.mean.abs()).fold(0.0f64, f64::max);
        grand = scale
            * moments
                .iter()
                .map(|m| (m.mean / scale) * (m.count as f64 / n as f64))
                .sum::<f64>()
                .clamp(-1.0, 1.0);
    }
    // Keep the two sums of squares independently in logarithmic form. A
    // common scale followed by ratio squaring can erase positive residual
    // variance under a large group contrast, before the tail is evaluated.
    let mut within = f64::NEG_INFINITY;
    let mut between = f64::NEG_INFINITY;
    for m in &moments {
        within = log_add(within, 2.0 * m.scale.ln() + m.normalized_squares.ln());
        between = log_add(
            between,
            (m.count as f64).ln() + 2.0 * log_difference(m.mean, grand),
        );
        cooperation.completed(1).await?;
    }
    let dfg = (k - 1) as f64;
    let dfe = (n - k) as f64;
    if moments
        .iter()
        .any(|m| !m.mean.is_finite() || !m.scale.is_finite())
    {
        within = f64::NAN;
        between = f64::NAN;
    }
    let log_msg = between - dfg.ln();
    let log_mse = within - dfe.ln();
    let f = (log_msg - log_mse).exp();
    // x=SSE/(SSE+SSB), independent of the displayed F or displayed SS.
    // Keep log(x) even if x itself cannot be represented: the beta tail can.
    let delta = between - within;
    let log_x = if delta > 0.0 {
        -delta - (-delta).exp().ln_1p()
    } else {
        -delta.exp().ln_1p()
    };
    let p =
        super::super::distribution_math::regularized_beta_from_log_x(log_x, dfe / 2.0, dfg / 2.0);
    let ssb = between.exp();
    let sse = within.exp();
    let empty = || Value::CharArray(CharArray::new_row(""));
    let row0 = ["Source", "SS", "df", "MS", "F", "Prob>F"]
        .map(|value| Value::CharArray(CharArray::new_row(value)));
    let rows = [
        row0,
        [
            Value::CharArray(CharArray::new_row("Groups")),
            Value::Num(ssb),
            Value::Num(dfg),
            Value::Num(log_msg.exp()),
            Value::Num(f),
            Value::Num(p),
        ],
        [
            Value::CharArray(CharArray::new_row("Error")),
            Value::Num(sse),
            Value::Num(dfe),
            Value::Num(log_mse.exp()),
            empty(),
            empty(),
        ],
        [
            Value::CharArray(CharArray::new_row("Total")),
            Value::Num(log_add(between, within).exp()),
            Value::Num((n - 1) as f64),
            empty(),
            empty(),
            empty(),
        ],
    ];
    // CellArray's physical 2-D payload is row-major (unlike Tensor storage).
    let cells = rows.into_iter().flatten().collect();
    let table = Value::Cell(CellArray::new(cells, 4, 6).map_err(|error| invalid(NAME, error))?);
    let mut stats = runmat_value::UnboundedStructValueBuilder::new();
    stats.insert("source", Value::CharArray(CharArray::new_row("anova1")));
    stats.insert(
        "gnames",
        Value::Cell(
            CellArray::new(
                data.names
                    .iter()
                    .map(|name| Value::CharArray(CharArray::new_row(name)))
                    .collect(),
                k,
                1,
            )
            .map_err(|error| invalid(NAME, error))?,
        ),
    );
    stats.insert(
        "n",
        Value::Tensor(
            Tensor::new(moments.iter().map(|m| m.count as f64).collect(), vec![1, k])
                .map_err(|error| invalid(NAME, error))?,
        ),
    );
    stats.insert(
        "means",
        Value::Tensor(
            Tensor::new(moments.iter().map(|m| m.mean).collect(), vec![1, k])
                .map_err(|error| invalid(NAME, error))?,
        ),
    );
    stats.insert("df", Value::Num(dfe));
    stats.insert("s", Value::Num((0.5 * log_mse).exp()));
    Ok(vec![Value::Num(p), table, Value::Struct(stats.finish())])
}

fn log_add(left: f64, right: f64) -> f64 {
    if left.is_nan() || right.is_nan() {
        return f64::NAN;
    }
    if left == f64::NEG_INFINITY {
        return right;
    }
    if right == f64::NEG_INFINITY {
        return left;
    }
    let high = left.max(right);
    let low = left.min(right);
    high + (low - high).exp().ln_1p()
}
fn log_difference(left: f64, right: f64) -> f64 {
    let difference = left - right;
    if difference.is_infinite() && left.is_finite() && right.is_finite() {
        let scale = left.abs().max(right.abs());
        scale.ln() + (left / scale - right / scale).abs().ln()
    } else {
        difference.abs().ln()
    }
}
