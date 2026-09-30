//! Shared interpretation of complete candidate observations. This feeds both
//! ordinary graph replanning and pre-graph candidate assembly.

use runmat_execution::{
    CandidateCostUnavailable, CandidateObservationEvidence, EstimateConfidence,
    ExecutionCandidateKind, ObservedCandidateDuration,
};

use super::{integer_sqrt, PlacementFeedbackSeries};

pub(super) fn from_series(
    series: &PlacementFeedbackSeries,
    tick: u64,
    kind: ExecutionCandidateKind,
) -> CandidateObservationEvidence {
    if tick.saturating_sub(series.latest_tick) > 1_024 {
        return CandidateObservationEvidence::Unavailable(
            CandidateCostUnavailable::StaleObservations,
        );
    }
    if series.failures >= 2 && kind.is_provider() {
        return CandidateObservationEvidence::Unavailable(
            CandidateCostUnavailable::RepeatedFailures {
                failed: series.failures,
            },
        );
    }
    if series.samples < 2 {
        return CandidateObservationEvidence::Unavailable(
            CandidateCostUnavailable::InsufficientObservations {
                successful: series.samples,
                failed: series.failures,
            },
        );
    }
    let Some(mean) = series.mean_ns() else {
        return CandidateObservationEvidence::Unavailable(
            CandidateCostUnavailable::InsufficientObservations {
                successful: series.samples,
                failed: series.failures,
            },
        );
    };
    let deviation = integer_sqrt(series.variance_ns2().unwrap_or(0));
    let variation_basis_points = if mean == 0 {
        u128::MAX
    } else {
        deviation.saturating_mul(10_000) / u128::from(mean)
    };
    let confidence = if series.samples >= 8 && variation_basis_points <= 1_000 {
        EstimateConfidence::High
    } else if variation_basis_points > 5_000 {
        EstimateConfidence::Low
    } else {
        EstimateConfidence::Medium
    };
    CandidateObservationEvidence::Observed(ObservedCandidateDuration {
        elapsed_ns: mean,
        confidence,
    })
}
