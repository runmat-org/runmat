use crate::contracts::FeaRunError;

use super::RandomResponseIntegrationPolicy;

pub(super) struct IntervalIntegration {
    pub values: Vec<f64>,
    pub evaluated_frequency_count: usize,
    pub refinement_level: u8,
    pub estimated_relative_error: f64,
}

/// Gauss-Kronrod 7/15 abscissae on [-1, 1] (non-negative half, QUADPACK `qk15`).
const KRONROD_NODES: [f64; 8] = [
    0.991_455_371_120_812_6,
    0.949_107_912_342_758_5,
    0.864_864_423_359_769_1,
    0.741_531_185_599_394_4,
    0.586_087_235_467_691_1,
    0.405_845_151_377_397_2,
    0.207_784_955_007_898_5,
    0.0,
];
const KRONROD_WEIGHTS: [f64; 8] = [
    0.022_935_322_010_529_22,
    0.063_092_092_629_978_55,
    0.104_790_010_322_250_18,
    0.140_653_259_715_525_92,
    0.169_004_726_639_267_9,
    0.190_350_578_064_785_4,
    0.204_432_940_075_298_9,
    0.209_482_141_084_727_83,
];
/// Gauss 7-point weights for Kronrod nodes 1, 3, 5 and the center.
const GAUSS_WEIGHTS: [f64; 4] = [
    0.129_484_966_168_869_7,
    0.279_705_391_489_276_7,
    0.381_830_050_505_118_9,
    0.417_959_183_673_469_4,
];
/// Hard bound on integrand evaluations per interval, independent of the authored depth.
const MAXIMUM_EVALUATIONS_PER_INTERVAL: usize = 1 << 21;

struct Segment {
    left: f64,
    right: f64,
    depth: u8,
    values: Vec<f64>,
    error: Vec<f64>,
}

fn kronrod_segment<F>(
    left: f64,
    right: f64,
    depth: u8,
    width: usize,
    evaluate: &mut F,
) -> Result<Segment, FeaRunError>
where
    F: FnMut(f64) -> Result<Vec<f64>, FeaRunError>,
{
    let center = 0.5 * (left + right);
    let half = 0.5 * (right - left);
    let mut kronrod = vec![0.0; width];
    let mut gauss = vec![0.0; width];
    let mut accumulate = |frequency: f64, node: usize| -> Result<(), FeaRunError> {
        let values = evaluate(frequency)?;
        if values.len() != width || values.iter().any(|value| !value.is_finite()) {
            return Err(FeaRunError::Solve(
                "random-response modal covariance density is malformed".into(),
            ));
        }
        for (index, value) in values.iter().enumerate() {
            kronrod[index] += KRONROD_WEIGHTS[node] * value;
            if node % 2 == 1 {
                gauss[index] += GAUSS_WEIGHTS[node / 2] * value;
            }
        }
        Ok(())
    };
    accumulate(center, 7)?;
    for node in 0..7 {
        let offset = half * KRONROD_NODES[node];
        accumulate(center - offset, node)?;
        accumulate(center + offset, node)?;
    }
    Ok(Segment {
        left,
        right,
        depth,
        values: kronrod.iter().map(|value| half * value).collect(),
        error: kronrod
            .iter()
            .zip(&gauss)
            .map(|(kronrod, gauss)| half * (kronrod - gauss).abs())
            .collect(),
    })
}

/// Largest error of a symmetric modal covariance relative to its own correlation scale.
///
/// Entry `(i, j)` is normalized by `sqrt(Q_ii Q_jj)`, so every recovered quantity
/// `phiᵀ Q phi` inherits the tolerance relative to `(Σ |phi_i| sqrt(Q_ii))²` regardless of
/// its absolute size. Modes with an exactly zero integral contribute exactly zero rows.
fn normalized_error(total: &[f64], error: &[f64], mode_count: usize) -> f64 {
    let mut worst = 0.0_f64;
    for row in 0..mode_count {
        for column in 0..mode_count {
            let index = row * mode_count + column;
            let scale = (total[row * mode_count + row].max(0.0)
                * total[column * mode_count + column].max(0.0))
            .sqrt();
            let relative = if scale > 0.0 {
                error[index] / scale
            } else if error[index] == 0.0 {
                0.0
            } else {
                f64::INFINITY
            };
            worst = worst.max(relative);
        }
    }
    worst
}

/// Globally adaptive Gauss-Kronrod integration of a modal covariance density over one interval.
///
/// The integrand is the real part of the `mode_count × mode_count` modal response covariance
/// density in row-major order. Bisection always targets the segment with the largest normalized
/// error; the authored refinement limit bounds bisection depth, so a resonance narrower than the
/// finest admitted segment fails closed instead of publishing an unconverged variance.
pub(super) fn integrate_interval<F>(
    left: f64,
    right: f64,
    policy: RandomResponseIntegrationPolicy,
    mode_count: usize,
    mut evaluate: F,
) -> Result<IntervalIntegration, FeaRunError>
where
    F: FnMut(f64) -> Result<Vec<f64>, FeaRunError>,
{
    if !left.is_finite() || !right.is_finite() || left < 0.0 || left >= right {
        return Err(FeaRunError::Solve(
            "random-response integration interval is invalid".into(),
        ));
    }
    let width = mode_count
        .checked_mul(mode_count)
        .filter(|width| *width > 0)
        .ok_or_else(|| FeaRunError::Solve("random-response modal dimension is invalid".into()))?;
    let middle = 0.5 * (left + right);
    let mut segments = vec![
        kronrod_segment(left, middle, 1, width, &mut evaluate)?,
        kronrod_segment(middle, right, 1, width, &mut evaluate)?,
    ];
    let mut evaluations = 30_usize;
    loop {
        let mut total = vec![0.0; width];
        let mut total_compensation = vec![0.0; width];
        let mut error = vec![0.0; width];
        for segment in &segments {
            compensated_add(&mut total, &mut total_compensation, &segment.values);
            for (sum, value) in error.iter_mut().zip(&segment.error) {
                *sum += value;
            }
        }
        let estimated = normalized_error(&total, &error, mode_count);
        if estimated <= policy.relative_variance_tolerance {
            return Ok(IntervalIntegration {
                values: total,
                evaluated_frequency_count: evaluations,
                refinement_level: segments
                    .iter()
                    .map(|segment| segment.depth)
                    .max()
                    .unwrap_or(1),
                estimated_relative_error: estimated,
            });
        }
        let (worst, _) = segments
            .iter()
            .enumerate()
            .map(|(index, segment)| (index, normalized_error(&total, &segment.error, mode_count)))
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .expect("an interval always has segments");
        let segment = &segments[worst];
        if segment.depth >= policy.maximum_refinement_levels
            || evaluations + 30 > MAXIMUM_EVALUATIONS_PER_INTERVAL
        {
            return Err(FeaRunError::Solve(format!(
                "random-response variance integration did not converge on [{left}, {right}] Hz: estimated relative error {estimated:.3e} exceeds {:.3e} at bisection depth {} ({evaluations} frequencies)",
                policy.relative_variance_tolerance, segment.depth
            )));
        }
        let segment = segments.swap_remove(worst);
        let center = 0.5 * (segment.left + segment.right);
        let depth = segment.depth + 1;
        segments.push(kronrod_segment(
            segment.left,
            center,
            depth,
            width,
            &mut evaluate,
        )?);
        segments.push(kronrod_segment(
            center,
            segment.right,
            depth,
            width,
            &mut evaluate,
        )?);
        evaluations += 30;
    }
}

pub(super) fn compensated_add(total: &mut [f64], compensation: &mut [f64], values: &[f64]) {
    for ((sum, correction), value) in total.iter_mut().zip(compensation).zip(values) {
        let corrected = value - *correction;
        let next = *sum + corrected;
        *correction = (next - *sum) - corrected;
        *sum = next;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptive_rule_integrates_quadratic_density() {
        let integrated = integrate_interval(
            0.0,
            1.0,
            RandomResponseIntegrationPolicy {
                modal_damping_ratio: 0.05,
                minimum_cumulative_mass_ratio: 0.9,
                relative_variance_tolerance: 1.0e-7,
                maximum_refinement_levels: 12,
            },
            1,
            |frequency| Ok(vec![frequency * frequency]),
        )
        .unwrap();
        assert!((integrated.values[0] - 1.0 / 3.0).abs() < 1.0e-7);
        assert!(integrated.evaluated_frequency_count >= 30);
    }

    #[test]
    fn under_resolved_peak_fails_without_a_dimensioned_absolute_escape_hatch() {
        let outcome = integrate_interval(
            0.0,
            1.0,
            RandomResponseIntegrationPolicy {
                modal_damping_ratio: 0.05,
                minimum_cumulative_mass_ratio: 0.0,
                relative_variance_tolerance: 1.0e-8,
                maximum_refinement_levels: 1,
            },
            1,
            |frequency| {
                let offset = (frequency - 0.5) / 0.01;
                Ok(vec![(-offset * offset).exp()])
            },
        );
        let error = match outcome {
            Ok(_) => panic!("an under-resolved peak must not publish integration evidence"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("did not converge"));
    }

    #[test]
    fn lightly_damped_resonance_integrates_to_the_closed_form_variance() {
        // Unit-input SDOF transfer |H|^2 over hertz; its integral over all frequency is
        // 1 / (8 zeta omega_n^3). The resonance sits at an interval end, as resonance splits
        // place it in the planner.
        let (natural_hz, zeta) = (81.3_f64, 0.01);
        let omega_n = std::f64::consts::TAU * natural_hz;
        let density = |frequency: f64| {
            let omega = std::f64::consts::TAU * frequency;
            let real = omega_n * omega_n - omega * omega;
            let imaginary = 2.0 * zeta * omega_n * omega;
            Ok(vec![1.0 / (real * real + imaginary * imaginary)])
        };
        let policy = RandomResponseIntegrationPolicy {
            modal_damping_ratio: zeta,
            minimum_cumulative_mass_ratio: 0.0,
            relative_variance_tolerance: 1.0e-9,
            maximum_refinement_levels: 24,
        };
        let below = integrate_interval(0.0, natural_hz, policy, 1, density).unwrap();
        let above = integrate_interval(natural_hz, 1.0e4 * natural_hz, policy, 1, density).unwrap();
        let exact = 1.0 / (8.0 * zeta * omega_n.powi(3));
        let tail = 1.0
            / (3.0 * (std::f64::consts::TAU * 1.0e4 * natural_hz).powi(3))
            / std::f64::consts::TAU;
        let total = below.values[0] + above.values[0];
        assert!(
            ((total + tail) - exact).abs() < 1.0e-7 * exact,
            "total={total} exact={exact}"
        );
    }
}
