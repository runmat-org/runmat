//! Box quartiles, whiskers and outlier classification.
use super::*;

pub(super) fn stats_for_values(
    label: String,
    values: Vec<f64>,
    whisker: f64,
) -> BuiltinResult<BoxStats> {
    let mut finite = values
        .into_iter()
        .filter(|value| !value.is_nan())
        .collect::<Vec<_>>();
    if finite.is_empty() {
        return Err(invalid(
            "boxplot: each plotted box must contain at least one non-NaN value",
        ));
    }
    finite.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let observation_count = finite.len();
    let q1 = percentile_sorted(&finite, 0.25);
    let median = percentile_sorted(&finite, 0.5);
    let q3 = percentile_sorted(&finite, 0.75);
    let iqr = q3 - q1;
    let (lower_fence, upper_fence) = if whisker == 0.0 {
        // An overflowing IQR still has exact zero-length whiskers at Q1/Q3.
        // Multiplying 0 by an infinite intermediate would produce NaN fences.
        (q1, q3)
    } else {
        (q1 - whisker * iqr, q3 + whisker * iqr)
    };
    let mut lower_whisker = None;
    let mut upper_whisker = None;
    let mut outliers = Vec::new();
    for value in finite {
        if value < lower_fence || value > upper_fence {
            outliers.push(value);
        } else {
            lower_whisker = Some(lower_whisker.map_or(value, |old: f64| old.min(value)));
            upper_whisker = Some(upper_whisker.map_or(value, |old: f64| old.max(value)));
        }
    }
    Ok(BoxStats {
        label,
        position: 0.0,
        width: 0.5,
        color: Vec4::ONE,
        q1,
        median,
        q3,
        lower_whisker: lower_whisker.unwrap_or(median),
        upper_whisker: upper_whisker.unwrap_or(median),
        outliers,
        observation_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_whisker_keeps_extreme_outlier_when_iqr_overflows() {
        let max = f64::MAX;
        let stats = stats_for_values(
            "A".into(),
            vec![-0.8 * max, -0.8 * max, 0.8 * max, 0.8 * max, max],
            0.0,
        )
        .unwrap();
        assert_eq!(stats.q1, -0.8 * max);
        assert_eq!(stats.q3, 0.8 * max);
        assert_eq!(stats.outliers, vec![max]);
    }
}

fn percentile_sorted(values: &[f64], p: f64) -> f64 {
    if values.len() == 1 {
        return values[0];
    }
    let pos = p * (values.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        values[lo]
    } else {
        let frac = pos - lo as f64;
        values[lo] * (1.0 - frac) + values[hi] * frac
    }
}
