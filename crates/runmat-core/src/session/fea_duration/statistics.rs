//! Pure projection of existing profile moments into display evidence. This
//! owns no observations and performs no hardware or kernel extrapolation.

const MAX_SAFE_WIRE_INTEGER: u64 = 9_007_199_254_740_991;

pub(super) struct DurationSummary {
    pub mean_millis: u64,
    pub lower_millis: u64,
    pub upper_millis: u64,
    pub sample_count: u64,
}

pub(super) fn summarize(
    samples: u64,
    failures: u64,
    total_elapsed_ns: u128,
    total_squared_elapsed_ns: u128,
) -> Option<DurationSummary> {
    if !(3..=MAX_SAFE_WIRE_INTEGER).contains(&samples) || failures != 0 {
        return None;
    }
    let count = samples as f64;
    let mean = total_elapsed_ns as f64 / count;
    let second_moment = total_squared_elapsed_ns as f64 / count;
    if !mean.is_finite()
        || mean <= 0.0
        || !second_moment.is_finite()
        || second_moment < mean * mean * (1.0 - 1.0e-12)
    {
        return None;
    }
    let variance = ((second_moment - mean * mean).max(0.0) * count / (count - 1.0)).max(0.0);
    let deviation = variance.sqrt();
    if !deviation.is_finite() || deviation / mean > 0.25 {
        return None;
    }
    // This is observed variability, not a confidence interval for an unseen
    // machine. The caller separately admits source, route, freshness and confidence.
    let mean_millis = (mean / 1_000_000.0).ceil().max(1.0);
    let lower = ((mean - 2.0 * deviation).max(1.0) / 1_000_000.0)
        .floor()
        .max(1.0);
    let upper = ((mean + 2.0 * deviation) / 1_000_000.0)
        .ceil()
        .max(mean_millis);
    if upper > MAX_SAFE_WIRE_INTEGER as f64 {
        return None;
    }
    Some(DurationSummary {
        mean_millis: mean_millis as u64,
        lower_millis: (lower as u64).min(mean_millis as u64),
        upper_millis: upper as u64,
        sample_count: samples,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn measured(millis: &[u64]) -> Option<DurationSummary> {
        let elapsed = millis
            .iter()
            .map(|n| u128::from(*n) * 1_000_000)
            .collect::<Vec<_>>();
        summarize(
            elapsed.len() as u64,
            0,
            elapsed.iter().sum(),
            elapsed.iter().map(|n| n * n).sum(),
        )
    }
    #[test]
    fn unknown_and_sparse_evidence_stays_absent() {
        for samples in [vec![], vec![1000], vec![1000, 1000]] {
            assert!(measured(&samples).is_none());
        }
    }
    #[test]
    fn stable_observations_produce_the_observed_mean() {
        let v = measured(&[1000, 1000, 1000]).unwrap();
        assert_eq!(
            (
                v.mean_millis,
                v.lower_millis,
                v.upper_millis,
                v.sample_count
            ),
            (1000, 1000, 1000, 3)
        );
    }
    #[test]
    fn variability_is_retained_in_the_range() {
        let v = measured(&[900, 1000, 1100]).unwrap();
        assert_eq!(v.mean_millis, 1000);
        assert!(v.lower_millis <= 800 && v.upper_millis >= 1200);
    }
    #[test]
    fn unstable_measurements_do_not_become_predictions() {
        assert!(measured(&[700, 1000, 1300]).is_none());
        assert!(measured(&[1000, 1000, 10000]).is_none());
    }
    #[test]
    fn failures_and_invalid_moments_do_not_make_a_number() {
        assert!(summarize(3, 1, 3_000_000_000, 3_000_000_000_000_000_000).is_none());
        assert!(summarize(3, 0, 3_000_000_000, 0).is_none());
        assert!(summarize(3, 0, 0, 0).is_none());
    }
    #[test]
    fn tiny_positive_times_round_without_zero_or_inverted_bounds() {
        let v = summarize(3, 0, 3, 3).unwrap();
        assert_eq!((v.mean_millis, v.lower_millis, v.upper_millis), (1, 1, 1));
    }
    #[test]
    fn wire_unsafe_counts_are_not_serialized_as_rounded_numbers() {
        assert!(summarize(u64::MAX, 0, u128::MAX, u128::MAX).is_none());
    }
    #[test]
    fn sample_variance_boundary_is_respected() {
        assert!(measured(&[750, 1000, 1250]).is_some());
        assert!(measured(&[749, 1000, 1251]).is_none());
    }
}
