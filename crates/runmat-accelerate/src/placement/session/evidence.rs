//! Shared interpretation of complete candidate observations. This feeds both
//! ordinary graph replanning and pre-graph candidate assembly.

use runmat_execution::{
    CandidateCostUnavailable, CandidateObservationEvidence, EstimateConfidence,
    ObservedCandidateDuration,
};

use super::{integer_sqrt, PlacementFeedbackSeries};

#[derive(Clone, Copy, Default)]
pub(super) struct FeedbackNumbers {
    pub samples: u64,
    pub failures: u64,
    pub total_elapsed_ns: u128,
    pub total_squared_elapsed_ns: u128,
    pub latest_tick: u64,
}

impl FeedbackNumbers {
    pub(super) fn from_series(series: &PlacementFeedbackSeries) -> Self {
        Self {
            samples: series.samples,
            failures: series.failures,
            total_elapsed_ns: series.total_elapsed_ns,
            total_squared_elapsed_ns: series.total_squared_elapsed_ns,
            latest_tick: series.latest_tick,
        }
    }

    pub(super) fn mean_ns(self) -> Option<u64> {
        (self.samples > 0).then(|| {
            u64::try_from(self.total_elapsed_ns / u128::from(self.samples)).unwrap_or(u64::MAX)
        })
    }

    pub(super) fn variance_ns2(self) -> Option<u128> {
        if self.samples < 2 {
            return None;
        }
        let mean = self.total_elapsed_ns / u128::from(self.samples);
        Some(
            (self.total_squared_elapsed_ns / u128::from(self.samples))
                .saturating_sub(mean.saturating_mul(mean)),
        )
    }

    pub(super) fn record(&mut self, elapsed: u64, succeeded: bool, tick: u64) -> bool {
        if succeeded {
            if self.samples >= super::MAX_FEEDBACK_OBSERVATIONS {
                self.samples = self.samples.div_ceil(2);
                self.total_elapsed_ns = self.total_elapsed_ns.div_ceil(2);
                self.total_squared_elapsed_ns = self.total_squared_elapsed_ns.div_ceil(2);
            }
            let elapsed = if self.samples >= 4 {
                let mean = self.mean_ns().unwrap_or(elapsed).max(1);
                elapsed.clamp((mean / 4).max(1), mean.saturating_mul(4))
            } else {
                elapsed.max(1)
            };
            self.samples = self.samples.saturating_add(1);
            self.total_elapsed_ns = self.total_elapsed_ns.saturating_add(u128::from(elapsed));
            self.total_squared_elapsed_ns = self
                .total_squared_elapsed_ns
                .saturating_add(u128::from(elapsed).saturating_mul(u128::from(elapsed)));
        } else {
            self.failures = self
                .failures
                .saturating_add(1)
                .min(super::MAX_FEEDBACK_OBSERVATIONS);
        }
        self.latest_tick = tick;
        !succeeded || self.samples <= 2 || self.samples.is_power_of_two()
    }
}

pub(super) fn from_series(
    series: &PlacementFeedbackSeries,
    tick: u64,
) -> CandidateObservationEvidence {
    from_numbers(FeedbackNumbers::from_series(series), series.kind, tick)
}

pub(super) fn from_numbers(
    numbers: FeedbackNumbers,
    kind: runmat_execution::ExecutionCandidateKind,
    tick: u64,
) -> CandidateObservationEvidence {
    if tick.saturating_sub(numbers.latest_tick) > 1_024 {
        return CandidateObservationEvidence::Unavailable(
            CandidateCostUnavailable::StaleObservations,
        );
    }
    if numbers.failures >= 2 && kind.is_provider() {
        return CandidateObservationEvidence::Unavailable(
            CandidateCostUnavailable::RepeatedFailures {
                failed: numbers.failures,
            },
        );
    }
    if numbers.samples < 2 {
        return CandidateObservationEvidence::Unavailable(
            CandidateCostUnavailable::InsufficientObservations {
                successful: numbers.samples,
                failed: numbers.failures,
            },
        );
    }
    let Some(mean) = numbers.mean_ns() else {
        return CandidateObservationEvidence::Unavailable(
            CandidateCostUnavailable::InsufficientObservations {
                successful: numbers.samples,
                failed: numbers.failures,
            },
        );
    };
    let deviation = integer_sqrt(numbers.variance_ns2().unwrap_or(0));
    let variation_basis_points = if mean == 0 {
        u128::MAX
    } else {
        deviation.saturating_mul(10_000) / u128::from(mean)
    };
    let confidence = if numbers.samples >= 8 && variation_basis_points <= 1_000 {
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
