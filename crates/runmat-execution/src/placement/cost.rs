use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EstimateConfidence {
    Prior,
    Low,
    Medium,
    High,
    Exact,
}

impl EstimateConfidence {
    pub const fn uncertainty_basis_points(self) -> u32 {
        match self {
            Self::Prior => 2_500,
            Self::Low => 1_500,
            Self::Medium => 500,
            Self::High => 100,
            Self::Exact => 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EstimateSource {
    StaticPrior,
    Calibration,
    Observation,
    Compiler,
    Provider,
    Synthetic,
}

/// Complete additive cost decomposition for one legal execution candidate.
///
/// Components are deliberately executor-neutral. Providers retain ownership of
/// kernel scheduling, while placement retains ownership of comparing complete
/// candidates and residency transitions.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionCostComponents {
    pub compile_or_prepare_ns: u64,
    pub upload_ns: u64,
    pub allocation_ns: u64,
    pub queue_ns: u64,
    pub execution_ns: u64,
    pub synchronization_ns: u64,
    pub download_ns: u64,
    pub downstream_ns: u64,
}

impl ExecutionCostComponents {
    pub fn checked_total_ns(self) -> Option<u64> {
        [
            self.compile_or_prepare_ns,
            self.upload_ns,
            self.allocation_ns,
            self.queue_ns,
            self.execution_ns,
            self.synchronization_ns,
            self.download_ns,
            self.downstream_ns,
        ]
        .into_iter()
        .try_fold(0_u64, u64::checked_add)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionCostEstimate {
    pub components: ExecutionCostComponents,
    /// Optional physical scratch bound supplied by the estimator. This field
    /// is not a substitute for the candidate's resource-admission evidence.
    pub scratch_bytes: Option<u64>,
    pub confidence: EstimateConfidence,
    pub source: EstimateSource,
}

/// Candidate assembly must not turn missing route evidence into a zero-cost
/// estimate. Only `Estimated` can enter a numeric placement graph.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CandidateCostEvidence {
    Estimated(ExecutionCostEstimate),
    Unavailable(CandidateCostUnavailable),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CandidateCostUnavailable {
    NoEstimate,
    UnrepresentableEstimate,
    NoObservations,
    InsufficientObservations { successful: u64, failed: u64 },
    RepeatedFailures { failed: u64 },
    StaleObservations,
}

impl CandidateCostEvidence {
    pub const fn from_estimate(estimate: Option<ExecutionCostEstimate>) -> Self {
        match estimate {
            Some(estimate) => Self::Estimated(estimate),
            None => Self::Unavailable(CandidateCostUnavailable::NoEstimate),
        }
    }

    pub const fn estimate(self) -> Option<ExecutionCostEstimate> {
        match self {
            Self::Estimated(estimate) => Some(estimate),
            Self::Unavailable(_) => None,
        }
    }

    /// The caller supplies separately admitted scratch evidence when known.
    /// A completed duration is a full-route observation, so its component is
    /// not split into guessed upload/kernel/download times.
    pub const fn from_observation(
        observation: CandidateObservationEvidence,
        scratch_bytes: Option<u64>,
    ) -> Self {
        match observation {
            CandidateObservationEvidence::Observed(duration) => {
                Self::Estimated(ExecutionCostEstimate {
                    components: ExecutionCostComponents {
                        compile_or_prepare_ns: 0,
                        upload_ns: 0,
                        allocation_ns: 0,
                        queue_ns: 0,
                        execution_ns: duration.elapsed_ns,
                        synchronization_ns: 0,
                        download_ns: 0,
                        downstream_ns: 0,
                    },
                    scratch_bytes,
                    confidence: duration.confidence,
                    source: EstimateSource::Observation,
                })
            }
            CandidateObservationEvidence::Unavailable(reason) => Self::Unavailable(reason),
        }
    }
}

/// Feedback measures completed wall time, not peak/scratch allocation or
/// individual transfer stages. Candidate assembly retains those independent
/// facts when combining this observation with its own legal route.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ObservedCandidateDuration {
    pub elapsed_ns: u64,
    pub confidence: EstimateConfidence,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CandidateObservationEvidence {
    Observed(ObservedCandidateDuration),
    Unavailable(CandidateCostUnavailable),
}

/// Preparation economics derived from two completed, comparable execution
/// observations. This is not an execution permit or a prediction of reuse.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PreparationPaybackEvidence {
    Unavailable(PreparationPaybackUnavailable),
    NonImproving {
        preparation_ns: u64,
        baseline_ns: u64,
        prepared_ns: u64,
        confidence: EstimateConfidence,
    },
    BreakEven {
        preparation_ns: u64,
        saving_per_invocation_ns: u64,
        invocations: u64,
        confidence: EstimateConfidence,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PreparationPaybackUnavailable {
    PreparationUnavailable,
    Baseline(CandidateCostUnavailable),
    Prepared(CandidateCostUnavailable),
    ArithmeticOverflow,
}

impl PreparationPaybackEvidence {
    /// The caller supplies exact paid preparation and observations from the
    /// same semantic boundary/current recipe. Warm execution costs retain no
    /// amortized preparation surcharge: that cost is paid only when preparing.
    pub fn from_observations(
        preparation_ns: Option<u64>,
        baseline: CandidateObservationEvidence,
        prepared: CandidateObservationEvidence,
    ) -> Self {
        let Some(preparation_ns) = preparation_ns else {
            return Self::Unavailable(PreparationPaybackUnavailable::PreparationUnavailable);
        };
        let baseline = match baseline {
            CandidateObservationEvidence::Observed(observed) => observed,
            CandidateObservationEvidence::Unavailable(reason) => {
                return Self::Unavailable(PreparationPaybackUnavailable::Baseline(reason))
            }
        };
        let prepared = match prepared {
            CandidateObservationEvidence::Observed(observed) => observed,
            CandidateObservationEvidence::Unavailable(reason) => {
                return Self::Unavailable(PreparationPaybackUnavailable::Prepared(reason))
            }
        };
        let Some(saving) = baseline
            .elapsed_ns
            .checked_sub(prepared.elapsed_ns)
            .filter(|saving| *saving > 0)
        else {
            return Self::NonImproving {
                preparation_ns,
                baseline_ns: baseline.elapsed_ns,
                prepared_ns: prepared.elapsed_ns,
                confidence: baseline.confidence.min(prepared.confidence),
            };
        };
        let Some(invocations) =
            (preparation_ns / saving).checked_add(u64::from(preparation_ns % saving != 0))
        else {
            return Self::Unavailable(PreparationPaybackUnavailable::ArithmeticOverflow);
        };
        Self::BreakEven {
            preparation_ns,
            saving_per_invocation_ns: saving,
            invocations,
            confidence: baseline.confidence.min(prepared.confidence),
        }
    }
}

impl ExecutionCostEstimate {
    pub fn checked_total_ns(self) -> Option<u64> {
        self.components.checked_total_ns()
    }

    pub fn checked_risk_adjusted_ns(self) -> Option<u64> {
        let total = self.checked_total_ns()?;
        let uncertainty = total
            .checked_mul(u64::from(self.confidence.uncertainty_basis_points()))?
            .checked_add(9_999)?
            / 10_000;
        total.checked_add(uncertainty)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preparation_payback_is_checked_unknown_aware_and_never_a_warm_surcharge() {
        let observed = |elapsed_ns| {
            CandidateObservationEvidence::Observed(ObservedCandidateDuration {
                elapsed_ns,
                confidence: EstimateConfidence::High,
            })
        };
        assert_eq!(
            PreparationPaybackEvidence::from_observations(Some(101), observed(100), observed(50)),
            PreparationPaybackEvidence::BreakEven {
                preparation_ns: 101,
                saving_per_invocation_ns: 50,
                invocations: 3,
                confidence: EstimateConfidence::High
            }
        );
        assert!(matches!(
            PreparationPaybackEvidence::from_observations(Some(1), observed(100), observed(100)),
            PreparationPaybackEvidence::NonImproving { .. }
        ));
        assert!(matches!(
            PreparationPaybackEvidence::from_observations(None, observed(100), observed(50)),
            PreparationPaybackEvidence::Unavailable(
                PreparationPaybackUnavailable::PreparationUnavailable
            )
        ));
        assert!(matches!(
            PreparationPaybackEvidence::from_observations(
                Some(1),
                CandidateObservationEvidence::Unavailable(
                    CandidateCostUnavailable::StaleObservations
                ),
                observed(50)
            ),
            PreparationPaybackEvidence::Unavailable(PreparationPaybackUnavailable::Baseline(
                CandidateCostUnavailable::StaleObservations
            ))
        ));
        assert!(matches!(
            PreparationPaybackEvidence::from_observations(Some(u64::MAX), observed(2), observed(1)),
            PreparationPaybackEvidence::BreakEven {
                invocations: u64::MAX,
                ..
            }
        ));
    }

    #[test]
    fn component_totals_and_uncertainty_are_checked() {
        let estimate = ExecutionCostEstimate {
            components: ExecutionCostComponents {
                execution_ns: 100,
                upload_ns: 20,
                ..ExecutionCostComponents::default()
            },
            scratch_bytes: Some(0),
            confidence: EstimateConfidence::Medium,
            source: EstimateSource::Synthetic,
        };
        assert_eq!(estimate.checked_total_ns(), Some(120));
        assert_eq!(estimate.checked_risk_adjusted_ns(), Some(126));

        let overflow = ExecutionCostComponents {
            execution_ns: u64::MAX,
            upload_ns: 1,
            ..ExecutionCostComponents::default()
        };
        assert_eq!(overflow.checked_total_ns(), None);
        assert_eq!(
            CandidateCostEvidence::from_estimate(None),
            CandidateCostEvidence::Unavailable(CandidateCostUnavailable::NoEstimate)
        );
        assert_eq!(
            CandidateCostEvidence::from_estimate(Some(estimate)).estimate(),
            Some(estimate)
        );
        let observed = CandidateCostEvidence::from_observation(
            CandidateObservationEvidence::Observed(ObservedCandidateDuration {
                elapsed_ns: 75,
                confidence: EstimateConfidence::Medium,
            }),
            Some(4_096),
        )
        .estimate()
        .unwrap();
        assert_eq!(observed.checked_total_ns(), Some(75));
        assert_eq!(observed.scratch_bytes, Some(4_096));
        assert_eq!(observed.source, EstimateSource::Observation);
        let unknown = CandidateCostEvidence::from_observation(
            CandidateObservationEvidence::Observed(ObservedCandidateDuration {
                elapsed_ns: 75,
                confidence: EstimateConfidence::Medium,
            }),
            None,
        )
        .estimate()
        .unwrap();
        assert_eq!(unknown.scratch_bytes, None);
        assert_eq!(unknown.checked_total_ns(), Some(75));

        let known_json = serde_json::to_value(estimate).unwrap();
        assert_eq!(known_json["scratch_bytes"], 0);
        let decoded: ExecutionCostEstimate = serde_json::from_value(known_json).unwrap();
        assert_eq!(decoded.scratch_bytes, Some(0));
        let unknown_json = serde_json::to_value(unknown).unwrap();
        assert!(unknown_json["scratch_bytes"].is_null());
    }
}
