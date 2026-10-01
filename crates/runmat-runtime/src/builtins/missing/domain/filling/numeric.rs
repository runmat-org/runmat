mod stable;

pub(in crate::builtins::missing::domain) enum Summary {
    Mean,
    Median,
}

pub(super) fn fill_summary_numeric(data: &mut [f64], lines: &FillLines, summary: Summary) {
    for line in lines.iter() {
        if !(0..line.length).any(|offset| data[line.start + offset * line.stride].is_nan()) {
            continue;
        }
        let values = (0..line.length)
            .map(|offset| data[line.start + offset * line.stride])
            .filter(|value| !value.is_nan())
            .collect();
        let replacement = summary_value(values, &summary);
        for offset in 0..line.length {
            let index = line.start + offset * line.stride;
            if data[index].is_nan() {
                data[index] = replacement;
            }
        }
    }
}

pub(in crate::builtins::missing::domain) fn summary_value(
    mut vals: Vec<f64>,
    summary: &Summary,
) -> f64 {
    if vals.is_empty() {
        return f64::NAN;
    }
    match summary {
        Summary::Mean => stable::mean(&vals),
        Summary::Median => {
            let mid = vals.len() / 2;
            let even = vals.len().is_multiple_of(2);
            let (lower, middle, _) = vals.select_nth_unstable_by(mid, f64::total_cmp);
            if even {
                let lower_middle = lower
                    .iter()
                    .copied()
                    .max_by(f64::total_cmp)
                    .expect("a nonempty even sample has a lower median");
                stable::midpoint(lower_middle, *middle)
            } else {
                *middle
            }
        }
    }
}

#[derive(Clone, Copy)]
pub(super) enum Neighbor {
    Previous,
    Next,
}

pub(super) fn fill_neighbor_numeric(data: &mut [f64], lines: &FillLines, dir: Neighbor) {
    for line in lines.iter() {
        fill_line_numeric(data, line.start, line.length, line.stride, dir);
    }
}

pub(super) fn fill_line_numeric(
    data: &mut [f64],
    start: usize,
    len: usize,
    step: usize,
    dir: Neighbor,
) {
    match dir {
        Neighbor::Previous => {
            let mut last = None;
            for i in 0..len {
                let idx = start + i * step;
                if data[idx].is_nan() {
                    if let Some(value) = last {
                        data[idx] = value;
                    }
                } else {
                    last = Some(data[idx]);
                }
            }
        }
        Neighbor::Next => {
            let mut next = None;
            for i in (0..len).rev() {
                let idx = start + i * step;
                if data[idx].is_nan() {
                    if let Some(value) = next {
                        data[idx] = value;
                    }
                } else {
                    next = Some(data[idx]);
                }
            }
        }
    }
}

pub(super) fn fill_nearest_numeric(data: &mut [f64], lines: &FillLines) {
    for line in lines.iter() {
        fill_nearest_line_numeric(data, line.start, line.length, line.stride);
    }
}

pub(super) fn fill_nearest_line_numeric(data: &mut [f64], start: usize, len: usize, step: usize) {
    let mut runs = super::runs::MissingRuns::new(len);
    while let Some(run) = runs.next(|position| data[start + position * step].is_nan()) {
        for position in run.start..run.end {
            if let Some(selected) = run.nearest_source(position) {
                data[start + position * step] = data[start + selected * step];
            }
        }
    }
}

pub(super) fn fill_linear_numeric(data: &mut [f64], lines: &FillLines) {
    for line in lines.iter() {
        fill_linear_line(data, line.start, line.length, line.stride);
    }
}

pub(super) fn fill_linear_line(data: &mut [f64], start: usize, len: usize, step: usize) {
    let mut i = 0;
    while i < len {
        let idx = start + i * step;
        if !data[idx].is_nan() {
            i += 1;
            continue;
        }
        let run_start = i;
        while i < len && data[start + i * step].is_nan() {
            i += 1;
        }
        let run_end = i;
        let prev = (run_start > 0).then(|| data[start + (run_start - 1) * step]);
        let next = (run_end < len).then(|| data[start + run_end * step]);
        match (prev, next) {
            (Some(a), Some(b)) if !a.is_nan() && !b.is_nan() => {
                let span = (run_end - run_start + 1) as f64;
                for (offset, pos) in (run_start..run_end).enumerate() {
                    data[start + pos * step] =
                        stable::interpolation(a, b, (offset + 1) as f64 / span);
                }
            }
            (Some(a), _) if !a.is_nan() => {
                for pos in run_start..run_end {
                    data[start + pos * step] = a;
                }
            }
            (_, Some(b)) if !b.is_nan() => {
                for pos in run_start..run_end {
                    data[start + pos * step] = b;
                }
            }
            _ => {}
        }
    }
}
use super::lines::FillLines;

#[cfg(test)]
mod tests {
    use super::{fill_linear_line, summary_value, Summary};

    #[test]
    fn mixed_signed_zero_midpoint_does_not_inherit_sorted_negative_zero() {
        let median = summary_value(vec![-0.0, 0.0], &Summary::Median);
        assert_eq!(median.to_bits(), 0.0f64.to_bits());
        let mut line = [-0.0, f64::NAN, 0.0];
        fill_linear_line(&mut line, 0, 3, 1);
        assert_eq!(line[1].to_bits(), 0.0f64.to_bits());
    }

    #[test]
    fn finite_summary_extremes_do_not_overflow() {
        for method in [Summary::Mean, Summary::Median] {
            assert_eq!(summary_value(vec![f64::MAX, f64::MAX], &method), f64::MAX);
            assert_eq!(summary_value(vec![-f64::MAX, f64::MAX], &method), 0.0);
            assert_eq!(
                summary_value(vec![f64::from_bits(1), f64::from_bits(1)], &method),
                f64::from_bits(1)
            );
            assert!(summary_value(vec![f64::INFINITY, f64::NEG_INFINITY], &method).is_nan());
            assert_eq!(
                summary_value(vec![f64::INFINITY, f64::INFINITY], &method),
                f64::INFINITY
            );
            assert!(summary_value(vec![-0.0, -0.0], &method).is_sign_negative());
        }
    }

    #[test]
    fn mean_keeps_small_residual_after_large_cancellation_in_any_order() {
        assert_eq!(
            summary_value(
                vec![-1e300, 2_f64.powi(53), 1e300, -0.25, -2_f64.powi(53)],
                &Summary::Mean
            ),
            -0.05
        );
        assert_eq!(
            summary_value(vec![2_f64.powi(51), -3.75, 4.0], &Summary::Mean),
            750_599_937_895_082.8
        );
        assert_eq!(
            summary_value(vec![1.0, 2_f64.powi(-53), 0.0], &Summary::Mean),
            f64::from_bits((1.0_f64 / 3.0).to_bits() + 1)
        );
        for values in [
            vec![f64::MAX, 1.0, -f64::MAX],
            vec![-f64::MAX, f64::MAX, 1.0],
            vec![1.0, -f64::MAX, f64::MAX],
        ] {
            assert_eq!(summary_value(values, &Summary::Mean), 1.0 / 3.0);
        }
        for values in [
            vec![f64::MAX, -f64::MAX, f64::MIN_POSITIVE],
            vec![f64::MIN_POSITIVE, f64::MAX, -f64::MAX],
        ] {
            assert_eq!(
                summary_value(values, &Summary::Mean),
                f64::MIN_POSITIVE / 3.0
            );
        }
        assert_eq!(
            summary_value(vec![f64::MAX, f64::MAX, -f64::MAX], &Summary::Mean),
            f64::MAX / 3.0
        );
        assert_eq!(
            summary_value(
                vec![f64::MAX, f64::MAX, -f64::MAX, 1.0, -f64::MAX],
                &Summary::Mean
            ),
            0.2
        );
        let mut subnormal_after_overflow = vec![f64::MAX, f64::MAX, -f64::MAX, -f64::MAX];
        subnormal_after_overflow.extend(std::iter::repeat_n(f64::from_bits(1), 5));
        assert_eq!(
            summary_value(subnormal_after_overflow, &Summary::Mean),
            f64::from_bits(1)
        );
    }

    #[test]
    fn linear_fill_keeps_finite_midpoints_and_strided_lines() {
        let mut adjacent_subnormals = [f64::from_bits(1), f64::NAN, f64::from_bits(2)];
        fill_linear_line(&mut adjacent_subnormals, 0, 3, 1);
        assert_eq!(adjacent_subnormals[1], f64::from_bits(2));
        let mut data = vec![-f64::MAX, 42.0, f64::NAN, 43.0, f64::MAX, 44.0];
        fill_linear_line(&mut data, 0, 3, 2);
        assert_eq!(data, [-f64::MAX, 42.0, 0.0, 43.0, f64::MAX, 44.0]);

        for endpoint in [f64::MAX, f64::from_bits(1), -0.0] {
            let mut line = [endpoint, f64::NAN, endpoint];
            fill_linear_line(&mut line, 0, 3, 1);
            assert_eq!(line[1], endpoint);
            if endpoint == 0.0 {
                assert!(line[1].is_sign_negative());
            }
        }
        let mut infinities = [f64::INFINITY, f64::NAN, f64::INFINITY];
        fill_linear_line(&mut infinities, 0, 3, 1);
        assert_eq!(infinities[1], f64::INFINITY);

        let mut opposite_infinities = [f64::NEG_INFINITY, f64::NAN, f64::INFINITY];
        fill_linear_line(&mut opposite_infinities, 0, 3, 1);
        assert!(opposite_infinities[1].is_nan());

        let mut long_run = [-f64::MAX, f64::NAN, f64::NAN, f64::MAX];
        fill_linear_line(&mut long_run, 0, 4, 1);
        assert!(long_run[1].is_finite() && long_run[2].is_finite());
        assert!(long_run[1] < 0.0 && long_run[2] > 0.0);
        assert!(((long_run[1] + long_run[2]) / f64::MAX).abs() < f64::EPSILON);
    }
}
