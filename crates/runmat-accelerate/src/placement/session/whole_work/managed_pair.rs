//! Managed complete-call choice for at most three already-admitted physical
//! routes. It shares the graph planner's exact single-route resource penalty
//! and host-margin policy without allocating a portable graph or probe.

use super::*;
use crate::placement::partition::single_candidate_resource_adjustment_ns;
use crate::placement::PlacementPolicy;

fn measured_total(item: &Admitted<'_>, request: WholeWorkPlacementRequestRef<'_>) -> Option<u64> {
    item.cost
        .estimate()?
        .checked_risk_adjusted_ns()?
        .checked_add(
            single_candidate_resource_adjustment_ns(
                CandidateResourceFacts {
                    location: item.candidate.execution_location,
                    output_residency: item.candidate.output_residency,
                    presentation: &item.candidate.presentation,
                    kind: item.candidate.kind,
                    demand: item.candidate.resources,
                    transactional_results: item.candidate.transactional_results,
                },
                request.resources,
                request.require_transactional_results,
            )
            .ok()?,
        )
}

fn direct(
    item: &Admitted<'_>,
    operation: PlacementOperationId,
    kind: WholeWorkDecisionKind,
    predicted_total_ns: Option<u64>,
) -> SelectionDraft {
    SelectionDraft::Direct {
        selected: CompactSelectedExecutionCandidate {
            node: operation,
            candidate: item.identity,
            kind: item.candidate.kind,
        },
        kind,
        predicted_total_ns,
        probe: None,
    }
}

pub(super) fn select(
    session: &PlacementSession,
    request: WholeWorkPlacementRequestRef<'_>,
    admitted: &[Admitted<'_>],
    operation: PlacementOperationId,
) -> Result<SelectionDraft, PlacementPlanError> {
    debug_assert!(!admitted.is_empty() && admitted.len() <= 3);
    let identity = request
        .signature
        .permit_identity()
        .expect("validated borrowed placement signature has a permit identity");
    let pool = session
        .managed_observations
        .as_ref()
        .expect("managed selection checked its installed observation backing");
    if let [item] = admitted {
        let kind = if matches!(
            request.constraint,
            WholeWorkConstraint::ProviderRequired { .. }
        ) {
            WholeWorkDecisionKind::ProviderRequired
        } else if matches!(item.cost, CandidateCostEvidence::Unavailable(_)) {
            if pool.has_active(identity, item.identity) {
                WholeWorkDecisionKind::Unmeasured
            } else {
                WholeWorkDecisionKind::Exploratory
            }
        } else {
            WholeWorkDecisionKind::Uncompared
        };
        return Ok(direct(
            item,
            operation,
            kind,
            item.cost
                .estimate()
                .and_then(|cost| cost.checked_total_ns()),
        ));
    }
    if matches!(
        request.constraint,
        WholeWorkConstraint::ProviderRequired { .. }
    ) || admitted
        .iter()
        .all(|item| item.candidate.kind.is_provider())
    {
        let item = route_choice(admitted, true).ok_or_else(no_legal_candidate)?;
        return Ok(direct(
            item,
            operation,
            if matches!(
                request.constraint,
                WholeWorkConstraint::ProviderRequired { .. }
            ) {
                WholeWorkDecisionKind::ProviderRequired
            } else {
                WholeWorkDecisionKind::Uncompared
            },
            item.cost
                .estimate()
                .and_then(|cost| cost.checked_total_ns()),
        ));
    }

    // The first sample for each candidate is a genuine source invocation.
    // Unlike the portable route, its in-flight marker is the precharged
    // managed observation slot, not a BTreeSet key and boxed ProbeLease.
    let available_unpriced = |item: &Admitted<'_>| {
        matches!(item.cost, CandidateCostEvidence::Unavailable(_))
            && !pool.has_active(identity, item.identity)
    };
    if let Some(item) = admitted
        .iter()
        .find(|item| !item.candidate.kind.is_provider() && available_unpriced(item))
        .or_else(|| admitted.iter().find(|item| available_unpriced(item)))
    {
        return Ok(direct(
            item,
            operation,
            WholeWorkDecisionKind::Exploratory,
            None,
        ));
    }

    let host = admitted
        .iter()
        .filter(|item| !item.candidate.kind.is_provider())
        .filter_map(|item| measured_total(item, request).map(|total| (item, total)))
        .min_by_key(|(item, total)| (*total, item.identity))
        .map(|(item, _)| item)
        .or_else(|| route_choice(admitted, false))
        .ok_or_else(no_legal_candidate)?;
    let host_total = measured_total(host, request);
    let provider = route_choice(admitted, true);
    let provider_total = provider.and_then(|item| measured_total(item, request));
    let (item, kind, predicted_total_ns) = match (host_total, provider, provider_total) {
        (Some(host_ns), None, _) => (host, WholeWorkDecisionKind::Compared, Some(host_ns)),
        (Some(host_ns), Some(provider), Some(provider_ns)) => {
            let required = PlacementPolicy::default()
                .required_improvement_ns(host_ns)
                .unwrap_or(u64::MAX);
            if provider_ns
                .checked_add(required)
                .is_some_and(|adjusted| adjusted <= host_ns)
            {
                (provider, WholeWorkDecisionKind::Compared, Some(provider_ns))
            } else {
                (host, WholeWorkDecisionKind::Compared, Some(host_ns))
            }
        }
        // A missing host comparison price does not authorize a provider
        // simply because it has a quote. Preserve the ordinary host route.
        _ => (host, WholeWorkDecisionKind::Unmeasured, host_total),
    };
    Ok(direct(item, operation, kind, predicted_total_ns))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placement::PlacementSessionConfig;
    use runmat_execution::{
        CandidateExecutionLocation, CandidateOutputPresentation, CandidateOutputResidency,
        CandidatePreparationState, CandidateResourceDemand, Digest, ExecutionCostComponents,
        ExecutionCostEstimate, PlacementExecutionAuthorityRevision, PlacementResourceSnapshot,
        PlacementRevision, PlacementSignature, ProviderResourceSnapshot, WholeWorkCandidate,
        WholeWorkConstraint, WholeWorkPlacementRequest,
    };
    use runmat_runtime::context::host_effect::ManagedHostEffectLedger;

    fn route(provider: bool, execution_ns: u64) -> WholeWorkCandidate {
        let device_id = 7;
        WholeWorkCandidate {
            kind: if provider {
                ExecutionCandidateKind::ProviderFusion
            } else {
                ExecutionCandidateKind::SharedRuntime
            },
            execution_location: if provider {
                CandidateExecutionLocation::Provider { device_id }
            } else {
                CandidateExecutionLocation::Host
            },
            preparation: CandidatePreparationState::Warm,
            output_residency: if provider {
                CandidateOutputResidency::Provider { device_id }
            } else {
                CandidateOutputResidency::Host
            },
            presentation: CandidateOutputPresentation::Direct,
            route_facts: Digest::sha256(if provider { b"gpu" } else { b"cpu" }),
            cost: CandidateCostEvidence::Estimated(ExecutionCostEstimate {
                components: ExecutionCostComponents {
                    execution_ns,
                    ..Default::default()
                },
                scratch_bytes: Some(8),
                confidence: runmat_execution::EstimateConfidence::Exact,
                source: runmat_execution::EstimateSource::Synthetic,
            }),
            resources: CandidateResourceDemand {
                cpu_millicores: if provider { 0 } else { 1_000 },
                retained_bytes: 16.into(),
                scratch_bytes: 8.into(),
                provider_host_workspace: None,
                provider_input_stages: None,
                queue_slots: u32::from(provider),
                peak_live: None,
            },
            transactional_results: false,
        }
    }

    fn request(host_ns: u64, provider_ns: u64) -> WholeWorkPlacementRequest {
        WholeWorkPlacementRequest {
            signature: PlacementSignature {
                region: None,
                operation: "stochastic_evolution".into(),
                runtime_facts: Digest::sha256(b"same source"),
                provider_provenance: None,
                graph_output_authority: None,
                revision: PlacementRevision {
                    program: None,
                    catalog: Digest::sha256(b"catalog"),
                    compiler: Digest::sha256(b"compiler"),
                    authority: PlacementExecutionAuthorityRevision::Provider(Digest::sha256(
                        b"provider",
                    )),
                    policy: Digest::sha256(b"policy"),
                },
            },
            constraint: WholeWorkConstraint::AnyLegal,
            candidates: vec![route(false, host_ns), route(true, provider_ns)],
            resources: PlacementResourceSnapshot {
                cpu_millicores_available: 1_000,
                memory_available_bytes: Some(1_024),
                cancellation_requested: false,
                providers: vec![ProviderResourceSnapshot {
                    device_id: 7,
                    capacity_bytes: Some(1_024),
                    scratch_available_bytes: Some(1_024),
                    queue_depth: Some(0),
                    queue_limit: Some(1),
                    epoch: 1,
                    ..Default::default()
                }],
                epoch: 1,
            },
            deterministic: true,
            require_transactional_results: false,
        }
    }

    #[test]
    fn managed_two_route_comparison_matches_portable_one_node_policy() {
        for (host_ns, provider_ns) in [(100_000, 20_000), (100_000, 98_000)] {
            let work = request(host_ns, provider_ns);
            let portable = PlacementSession::default()
                .select_whole_work_borrowed(work.as_ref())
                .expect("portable comparison");
            let ledger = ManagedHostEffectLedger::new(32 * 1024);
            let managed = PlacementSession::new_with_managed_observations(
                PlacementSessionConfig::default(),
                &ledger,
                super::super::super::ManagedPlacementCapacity {
                    in_flight_slots: 1,
                    feedback_series: 2,
                    signature_bytes_per_series: 1024,
                },
            )
            .expect("precharged observation backing");
            let selected = managed
                .select_whole_work_borrowed_managed(work.as_ref())
                .expect("managed comparison");
            assert_eq!(selected.kind, WholeWorkDecisionKind::Compared);
            assert_eq!(selected.selected.portable(), portable.decision.selected);
            assert_eq!(
                selected.predicted_total_ns,
                portable.decision.predicted_total_ns
            );
        }
    }

    #[test]
    fn managed_three_host_routes_compare_without_spilling_or_zeroing_unknown_demand() {
        for winner in 0..3 {
            let mut work = request(100_000, 20_000);
            work.resources.providers.clear();
            work.candidates = (0..3)
                .map(|index| {
                    let mut item = route(false, if index == winner { 1_000 } else { 100_000 });
                    item.kind = match index {
                        0 => ExecutionCandidateKind::SharedRuntime,
                        1 => ExecutionCandidateKind::GenericNativeCpu,
                        _ => ExecutionCandidateKind::SpecializedNativeCpu,
                    };
                    item.route_facts = Digest::sha256([index as u8]);
                    item
                })
                .collect();
            let expected = work.candidates[winner].selected(&work.signature.operation);
            let ledger = ManagedHostEffectLedger::new(64 * 1024);
            let managed = PlacementSession::new_with_managed_observations(
                PlacementSessionConfig::default(),
                &ledger,
                super::super::super::ManagedPlacementCapacity {
                    in_flight_slots: 1,
                    feedback_series: 3,
                    signature_bytes_per_series: 1024,
                },
            )
            .expect("precharged three-route backing");
            let selected = managed
                .select_whole_work_borrowed_managed(work.as_ref())
                .expect("three certified routes fit");
            assert_eq!(selected.kind, WholeWorkDecisionKind::Compared);
            assert_eq!(selected.selected.portable(), expected);
            drop(selected);
            for item in &mut work.candidates {
                item.resources.scratch_bytes = runmat_execution::ResourceByteBound::Unavailable(
                    runmat_execution::ResourceBoundUnavailable::NoScratchBound,
                );
            }
            assert_eq!(
                managed
                    .select_whole_work_borrowed_managed(work.as_ref())
                    .err()
                    .expect("unknown demand cannot masquerade as zero")
                    .code,
                "RunMat:Placement:InsufficientResourceEvidence"
            );
            assert!(managed
                .try_select_whole_work_borrowed_managed(work.as_ref())
                .expect("typed decline is not an execution error")
                .is_none());
        }
    }
}
