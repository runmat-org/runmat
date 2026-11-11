use super::{ModalCandidate, MODAL_BOUNDARY_PROBE_WIDTH};

pub(super) struct ResolvedTruncationBoundary {
    pub(super) retained: Vec<ModalCandidate>,
    pub(super) probe_count: usize,
    pub(super) probe_frequency_hz: Option<f64>,
}

pub(super) fn resolve_truncation_boundary(
    candidates: &[ModalCandidate],
    requested_count: usize,
    tolerance: f64,
    exhausted: bool,
) -> Option<ResolvedTruncationBoundary> {
    if candidates.len() < requested_count {
        return exhausted.then(|| ResolvedTruncationBoundary {
            retained: candidates.to_vec(),
            probe_count: 0,
            probe_frequency_hz: None,
        });
    }
    let mut ordered = candidates.to_vec();
    ordered.sort_by(|left, right| left.0.total_cmp(&right.0));
    let mut retained_count = requested_count;
    while retained_count < ordered.len()
        && crate::pipeline::modal_governance::relative_frequency_separation(
            ordered[retained_count - 1].0,
            ordered[retained_count].0,
        ) <= tolerance
    {
        retained_count += 1;
    }
    let probe_count = ordered.len() - retained_count;
    if !exhausted && probe_count < MODAL_BOUNDARY_PROBE_WIDTH {
        return None;
    }
    let probe_frequency_hz = ordered.get(retained_count).map(|candidate| candidate.0);
    ordered.truncate(retained_count);
    Some(ResolvedTruncationBoundary {
        retained: ordered,
        probe_count,
        probe_frequency_hz,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(frequency_hz: f64) -> ModalCandidate {
        (frequency_hz, vec![frequency_hz], 0.0, 0.0)
    }

    #[test]
    fn boundary_resolution_orders_a_nonmonotonic_candidate_stream() {
        let candidates = [10.0, 1.0, 2.0, 6.0, 3.0, 5.0]
            .into_iter()
            .map(candidate)
            .collect::<Vec<_>>();
        let resolved = resolve_truncation_boundary(&candidates, 2, 1.0e-6, false).unwrap();
        assert_eq!(
            resolved
                .retained
                .iter()
                .map(|candidate| candidate.0)
                .collect::<Vec<_>>(),
            [1.0, 2.0]
        );
        assert_eq!(resolved.probe_count, 4);
        assert_eq!(resolved.probe_frequency_hz, Some(3.0));
    }

    #[test]
    fn boundary_resolution_keeps_a_complete_cluster() {
        let candidates = [8.0, 1.0, 2.0, 2.000_001, 4.0, 6.0, 10.0]
            .into_iter()
            .map(candidate)
            .collect::<Vec<_>>();
        let resolved = resolve_truncation_boundary(&candidates, 2, 1.0e-6, false).unwrap();
        assert_eq!(resolved.retained.len(), 3);
        assert_eq!(resolved.probe_frequency_hz, Some(4.0));
    }
}
