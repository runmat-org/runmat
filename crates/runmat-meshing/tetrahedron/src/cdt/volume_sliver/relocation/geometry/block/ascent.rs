//! A local linear maximin model includes all corner constraints and their slack.
//! Its bounded convex dual supplies proposals, never acceptance authority.
use super::*;
mod line;
use line::NormLine;
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
#[derive(Clone)]
struct Sample {
    value: f64,
    gradient: Vec<f64>,
}
impl PreparedBlock {
    pub(super) fn ascent_direction(
        &self,
        points: &[[f64; 3]],
        radius: f64,
    ) -> Result<Option<Vec<f64>>, DelaunayVolumeSliverError> {
        let mut samples = Vec::with_capacity(4 * self.cells.len());
        for cell in &self.cells {
            let positions = std::array::from_fn(|i| {
                cell.slots[i].map_or(cell.nodes[i].coordinates_m.canonical_value(), |p| points[p])
            });
            let Some(gradients) = cell.metric.corner_gradients(positions).map_err(|cause| {
                error(
                    DelaunayVolumeSliverErrorKind::Accounting(cause),
                    "block ascent numerical accounting failed",
                )
            })?
            else {
                return Ok(None);
            };
            for sample in gradients {
                let mut gradient = vec![0.; 3 * points.len()];
                for (i, slot) in cell.slots.iter().enumerate() {
                    if let Some(p) = slot {
                        for axis in 0..3 {
                            gradient[3 * p + axis] += sample.gradients[i][axis];
                        }
                    }
                }
                samples.push(Sample {
                    value: sample.value,
                    gradient,
                });
            }
        }
        Ok(proposal(&samples, radius))
    }
}
fn proposal(samples: &[Sample], radius: f64) -> Option<Vec<f64>> {
    let minimum = samples.iter().map(|s| s.value).min_by(f64::total_cmp)?;
    let norm = samples
        .iter()
        .map(|s| dot(&s.gradient, &s.gradient).sqrt())
        .fold(0., f64::max);
    if !norm.is_finite() || norm <= 0. || !radius.is_finite() || radius <= 0. {
        return None;
    }
    let normalized = samples
        .iter()
        .map(|sample| Sample {
            value: (sample.value - minimum) / (radius * norm),
            gradient: sample.gradient.iter().map(|v| v / norm).collect(),
        })
        .collect::<Vec<_>>();
    if normalized.iter().any(|s| !s.value.is_finite()) {
        return None;
    }
    let first = normalized
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.value.total_cmp(&b.value))?
        .0;
    let mut weights = vec![0.; normalized.len()];
    weights[first] = 1.;
    let mut mean = normalized[first].gradient.clone();
    let mut slack = normalized[first].value;
    let mut best = None;
    // Each constraint moves at most one cell's free corners. Sparse rows keep
    // the dominant per-iteration scan proportional to those corners, in index
    // order; omitted exact-zero products cannot change a derivative value.
    let values = normalized.iter().map(|s| s.value).collect::<Vec<_>>();
    let mut row_ends = Vec::with_capacity(normalized.len());
    let mut entries = Vec::new();
    for sample in &normalized {
        entries.extend(
            sample
                .gradient
                .iter()
                .enumerate()
                .filter(|(_, value)| **value != 0.)
                .map(|(index, value)| (index, *value)),
        );
        row_ends.push(entries.len());
    }
    let mut derivatives = Vec::with_capacity(normalized.len());
    // The ball-constrained linear maximin dual is |sum(lambda*g)| +
    // sum(lambda*slack). Smooth only the zero-norm derivative at roundoff scale;
    // every proposed direction remains inside the original geometric radius.
    for _ in 0..MAXIMUM_CAVITY_SMOOTHING_ITERATIONS {
        let length = dot(&mean, &mean).sqrt().hypot(64. * f64::EPSILON);
        let direction = mean.iter().map(|v| v / length).collect::<Vec<_>>();
        // One linear-model value per constraint per iteration; the minimum,
        // entering and away selections below all read this vector.
        derivatives.clear();
        let mut row_start = 0;
        derivatives.extend(row_ends.iter().zip(&values).map(|(&row_end, value)| {
            let row = &entries[row_start..row_end];
            row_start = row_end;
            value
                + row
                    .iter()
                    .map(|(index, gradient)| gradient * direction[*index])
                    .sum::<f64>()
        }));
        let predicted = derivatives.iter().copied().fold(f64::INFINITY, f64::min);
        if predicted > 64. * f64::EPSILON
            && best.as_ref().is_none_or(|(value, _)| predicted > *value)
        {
            best = Some((predicted, direction.clone()));
        }
        let next_index = derivatives
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.total_cmp(b))?
            .0;
        let next = &normalized[next_index];
        let mut delta = next
            .gradient
            .iter()
            .zip(&mean)
            .map(|(a, b)| a - b)
            .collect::<Vec<_>>();
        let mut slack_delta = next.value - slack;
        let line = NormLine::prepare(&mean, &delta, slack_delta);
        let mut step = line.minimum_step(1.);
        let mut away_index = None;
        // Transfer weight away from a stale active constraint when its exact
        // one-dimensional dual decrease exceeds the ordinary convex step.
        // The ordinary step always remains available under the same work cap.
        let away = derivatives
            .iter()
            .enumerate()
            .filter(|(index, _)| weights[*index] > 0.)
            .max_by(|(_, a), (_, b)| a.total_cmp(b))?
            .0;
        if away != next_index {
            let pair_delta = next
                .gradient
                .iter()
                .zip(&normalized[away].gradient)
                .map(|(a, b)| a - b)
                .collect::<Vec<_>>();
            let pair_slack = next.value - normalized[away].value;
            let pair_line = NormLine::prepare(&mean, &pair_delta, pair_slack);
            let pair_step = pair_line.minimum_step(weights[away]);
            if pair_line.value(pair_step) < line.value(step) {
                delta = pair_delta;
                slack_delta = pair_slack;
                step = pair_step;
                away_index = Some(away);
            }
        }
        if step <= 0. {
            break;
        }
        if let Some(away) = away_index {
            weights[away] -= step;
        } else {
            for weight in &mut weights {
                *weight *= 1. - step;
            }
        }
        weights[next_index] += step;
        mean = mean.iter().zip(&delta).map(|(a, b)| a + step * b).collect();
        slack += step * slack_delta;
    }
    best.map(|(_, direction)| direction)
}

#[cfg(test)]
mod tests {
    use super::*;
    // witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
    //   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
    //   boundary: The local linear maximin proposal includes constraint slack, improves competing
    //   affine constraints within its radius, rejects stationary or fixed bottlenecks, and preserves
    //   common scaling. The model supplies proposals only; independent complete geometry and current
    //   scoped quality decide every accepted edit. This witness does not assert original-study
    //   acceptance or measured speedup.
    #[test]
    fn linear_maximin_bundle_accounts_for_nearby_constraints_and_stationarity() {
        let sample = |value, x, y| Sample {
            value,
            gradient: vec![x, y, 0., 0., 0., 0.],
        };
        let samples = [sample(1., 1., 0.), sample(1.01, -1., 1.)];
        let step = proposal(&samples, 1.).unwrap();
        assert!(samples
            .iter()
            .all(|s| s.value + dot(&s.gradient, &step) > 1.));
        assert!(dot(&step, &step) <= 1.);
        assert!(proposal(&[sample(1., 1., 0.), sample(1., -1., 0.)], 1.).is_none());
        assert!(proposal(&[sample(1., 0., 0.), sample(2., 1., 0.)], 1.).is_none());
        let rescaled = samples.clone().map(|s| Sample {
            value: s.value * 1e3,
            gradient: s.gradient.iter().map(|v| v * 1e3).collect(),
        });
        let other = proposal(&rescaled, 1.).unwrap();
        for i in 0..6 {
            assert!((step[i] - other[i]).abs() < 1e-12);
        }
    }
    #[test]
    fn multi_constraint_model_reaches_the_independent_balanced_optimum_and_rejects_stationarity() {
        let samples = (0..3)
            .map(|axis| {
                let mut gradient = vec![0.; 3];
                gradient[axis] = 1.;
                Sample {
                    value: 1.,
                    gradient,
                }
            })
            .collect::<Vec<_>>();
        let direction = proposal(&samples, 1.).unwrap();
        let gain = samples
            .iter()
            .map(|sample| dot(&sample.gradient, &direction))
            .fold(f64::INFINITY, f64::min);
        assert!((gain - 1. / 3_f64.sqrt()).abs() < 1e-10);
        assert!(dot(&direction, &direction) <= 1. + 1e-14);
        let stationary = [vec![1., 0.], vec![0., 1.], vec![-1., -1.]].map(|gradient| Sample {
            value: 1.,
            gradient,
        });
        assert!(proposal(&stationary, 1.).is_none());
    }
    #[test]
    fn cluster_model_can_advance_a_constraint_outside_the_first_neighbor_pair() {
        let mut gradient = vec![0.; 9];
        gradient[7] = 1.;
        let step = proposal(
            &[Sample {
                value: 1.,
                gradient,
            }],
            1.,
        )
        .unwrap();
        assert_eq!(step.len(), 9);
        assert!(step[7] > 0.99);
        assert!(step[..6].iter().all(|v| *v == 0.));
    }
}
