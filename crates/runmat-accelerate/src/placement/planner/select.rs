use runmat_execution::ExecutionCandidateDescriptor;

use super::PlacementPolicy;

/// Compare complete, already-feasible candidates. Provider feasibility is
/// intentionally normalized before this function: profitability must never
/// make an unsupported candidate executable.
pub(crate) fn select_candidate(
    candidates: impl IntoIterator<Item = ExecutionCandidateDescriptor>,
    policy: PlacementPolicy,
) -> Option<ExecutionCandidateDescriptor> {
    let mut legal = Vec::new();
    for candidate in candidates {
        if candidate.validate().is_err() {
            continue;
        }
        let Some(risk_adjusted_ns) = candidate.cost.checked_risk_adjusted_ns() else {
            continue;
        };
        legal.push((candidate, risk_adjusted_ns));
    }
    legal.sort_by(|(left, left_cost), (right, right_cost)| {
        left_cost
            .cmp(right_cost)
            .then_with(|| left.identity.cmp(&right.identity))
    });
    let best_cpu = legal
        .iter()
        .filter(|(candidate, _)| !candidate.kind.is_provider())
        .min_by_key(|(_, cost)| *cost);
    let best_provider = legal
        .iter()
        .filter(|(candidate, _)| candidate.kind.is_provider())
        .min_by_key(|(_, cost)| *cost);
    let (selected, _) = match (best_cpu, best_provider) {
        (Some(cpu), Some(provider)) => {
            let required = policy.required_improvement_ns(cpu.1).unwrap_or(u64::MAX);
            if provider
                .1
                .checked_add(required)
                .is_some_and(|cost| cost <= cpu.1)
            {
                provider.clone()
            } else {
                cpu.clone()
            }
        }
        (Some(cpu), None) => cpu.clone(),
        (None, Some(provider)) => provider.clone(),
        (None, None) => return None,
    };
    Some(selected)
}

#[cfg(test)]
mod tests {
    use runmat_execution::{
        CandidateExecutionLocation, CandidateOutputResidency, CandidatePreparationState,
        EstimateConfidence, EstimateSource, ExecutionCandidateKind, ExecutionCostComponents,
        ExecutionCostEstimate,
    };

    use super::*;

    fn candidate(
        identity: &str,
        kind: ExecutionCandidateKind,
        execution_ns: u64,
        upload_ns: u64,
    ) -> ExecutionCandidateDescriptor {
        ExecutionCandidateDescriptor {
            identity: identity.into(),
            region: None,
            kind,
            execution_location: if kind.is_provider() {
                CandidateExecutionLocation::Provider { device_id: 1 }
            } else {
                CandidateExecutionLocation::Host
            },
            preparation: CandidatePreparationState::Warm,
            cost: ExecutionCostEstimate {
                components: ExecutionCostComponents {
                    execution_ns,
                    upload_ns,
                    ..ExecutionCostComponents::default()
                },
                scratch_bytes: 0,
                confidence: EstimateConfidence::Exact,
                source: EstimateSource::Synthetic,
            },
            output_residency: if kind.is_provider() {
                CandidateOutputResidency::Provider { device_id: 1 }
            } else {
                CandidateOutputResidency::Host
            },
            guards: Vec::new(),
        }
    }

    #[test]
    fn transfer_dominated_provider_loses_to_cpu() {
        let decision = select_candidate(
            [
                candidate("cpu", ExecutionCandidateKind::GenericNativeCpu, 40_000, 0),
                candidate(
                    "gpu",
                    ExecutionCandidateKind::ProviderOperation,
                    5_000,
                    80_000,
                ),
            ],
            PlacementPolicy::default(),
        )
        .unwrap();
        assert_eq!(decision.identity, "cpu");
    }

    #[test]
    fn resident_provider_chain_wins_on_complete_cost() {
        let decision = select_candidate(
            [
                candidate("cpu", ExecutionCandidateKind::GenericNativeCpu, 80_000, 0),
                candidate("gpu", ExecutionCandidateKind::ProviderFusion, 20_000, 0),
            ],
            PlacementPolicy::default(),
        )
        .unwrap();
        assert_eq!(decision.identity, "gpu");
    }

    #[test]
    fn uncertainty_and_ties_prefer_cpu() {
        let decision = select_candidate(
            [
                candidate("gpu", ExecutionCandidateKind::ProviderOperation, 10_000, 0),
                candidate("cpu", ExecutionCandidateKind::GenericNativeCpu, 10_000, 0),
            ],
            PlacementPolicy::default(),
        )
        .unwrap();
        assert_eq!(decision.identity, "cpu");
    }

    #[test]
    fn provider_must_clear_absolute_and_relative_margins() {
        let policy = PlacementPolicy {
            absolute_margin_ns: 5_000,
            relative_margin_basis_points: 1_000,
        };
        let close = select_candidate(
            [
                candidate("cpu", ExecutionCandidateKind::GenericNativeCpu, 100_000, 0),
                candidate("gpu", ExecutionCandidateKind::ProviderOperation, 94_000, 0),
            ],
            policy,
        )
        .unwrap();
        assert_eq!(close.identity, "cpu");

        let clear = select_candidate(
            [
                candidate("cpu", ExecutionCandidateKind::GenericNativeCpu, 100_000, 0),
                candidate("gpu", ExecutionCandidateKind::ProviderOperation, 89_000, 0),
            ],
            policy,
        )
        .unwrap();
        assert_eq!(clear.identity, "gpu");
    }
}
