//! Prepare the orthogonal decomposition once for a scalar convex line search.
//! Repeated derivative queries do not revisit or allocate coordinate vectors.
pub(super) struct NormLine {
    delta_norm: f64,
    parallel: f64,
    perpendicular: f64,
    slack_delta: f64,
}
impl NormLine {
    pub(super) fn prepare(mean: &[f64], delta: &[f64], slack_delta: f64) -> Self {
        let delta_norm = delta
            .iter()
            .fold(0_f64, |length, value| length.hypot(*value));
        if delta_norm == 0. {
            return Self {
                delta_norm,
                parallel: 0.,
                perpendicular: mean
                    .iter()
                    .fold(0_f64, |length, value| length.hypot(*value)),
                slack_delta,
            };
        }
        let parallel = mean
            .iter()
            .zip(delta)
            .map(|(a, b)| a * (b / delta_norm))
            .sum::<f64>();
        let perpendicular = mean.iter().zip(delta).fold(0_f64, |length, (a, b)| {
            length.hypot(a - parallel * (b / delta_norm))
        });
        Self {
            delta_norm,
            parallel,
            perpendicular,
            slack_delta,
        }
    }
    pub(super) fn value(&self, step: f64) -> f64 {
        let axial = self.parallel + step * self.delta_norm;
        axial.hypot(self.perpendicular).hypot(64. * f64::EPSILON) + step * self.slack_delta
    }
    pub(super) fn minimum_step(&self, maximum: f64) -> f64 {
        if self.derivative(0.) >= -64. * f64::EPSILON {
            return 0.;
        }
        if self.derivative(maximum) <= 0. {
            return maximum;
        }
        let mut lower = 0.;
        let mut upper = maximum;
        for _ in 0..48 {
            let middle = 0.5 * (lower + upper);
            if self.derivative(middle) < 0. {
                lower = middle;
            } else {
                upper = middle;
            }
        }
        0.5 * (lower + upper)
    }
    pub(super) fn derivative(&self, step: f64) -> f64 {
        let axial = self.parallel + step * self.delta_norm;
        self.delta_norm * (axial / axial.hypot(self.perpendicular).hypot(64. * f64::EPSILON))
            + self.slack_delta
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn reference(mean: &[f64], delta: &[f64], step: f64, slack: f64) -> f64 {
        let (product, length) =
            mean.iter()
                .zip(delta)
                .fold((0., 0_f64), |(product, length), (a, b)| {
                    let value = a + step * b;
                    (product + value * b, length.hypot(value))
                });
        product / length.hypot(64. * f64::EPSILON) + slack
    }
    #[test]
    fn convex_line_value_and_minimum_match_independent_analytic_cases() {
        let constant_direction = NormLine::prepare(&[3., 4.], &[0., 0.], -0.2);
        assert!((constant_direction.value(0.5) - 4.9).abs() < 1e-14);
        assert_eq!(constant_direction.minimum_step(1.), 1.);
        let symmetric = NormLine::prepare(&[1., 1.], &[-2., 0.], 0.);
        assert!((symmetric.minimum_step(1.) - 0.5).abs() < 1e-13);
        assert!((symmetric.value(0.5) - 1.).abs() < 1e-14);
        assert_eq!(
            NormLine::prepare(&[1., 1.], &[-2., 0.], 2.).minimum_step(1.),
            0.
        );
        assert_eq!(
            NormLine::prepare(&[1., 1.], &[-2., 0.], -2.).minimum_step(1.),
            1.
        );
    }
    #[test]
    fn prepared_norm_line_matches_independent_vector_derivatives_including_cancellation() {
        for (mean, delta) in [
            (vec![0.3, -0.2, 0.7], vec![-0.4, 0.8, 0.1]),
            (vec![1., 0., 0.], vec![-1., 0., 0.]),
            (vec![1., 1e-9, 0.], vec![-1., 0., 0.]),
            (vec![1., 2., 3.], vec![0., 0., 0.]),
            (
                (0..60).map(|i| (f64::from(i) * 0.17).sin()).collect(),
                (0..60).map(|i| (f64::from(i) * 0.31).cos()).collect(),
            ),
        ] {
            for slack in [0., 0.03, -0.03] {
                let line = NormLine::prepare(&mean, &delta, slack);
                for step in [0., 0.1, 0.5, 1. - 1e-12, 1., 1. + 1e-12] {
                    let expected = reference(&mean, &delta, step, slack);
                    let actual = line.derivative(step);
                    assert!(
                        (actual - expected).abs() <= 3e-12 * expected.abs().max(1.),
                        "{actual} != {expected}"
                    );
                }
            }
        }
    }
}
