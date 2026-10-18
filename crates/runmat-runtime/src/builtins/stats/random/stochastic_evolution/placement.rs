//! One pre-effect whole-call choice for the already prepared source program.
//! Resource arithmetic belongs to resources.rs and profitability to the
//! installed placement service; this module does not invent either price.

use super::host_readback::StochasticHostReadback;
use super::planning::{profile, StochasticProviderPlan};
use super::resources::{StochasticHostResources, StochasticProviderResources};
use super::source_eligibility::StochasticPreparedSource;
use crate::builtins::common::random::gpu_like::{host_random_mode, ExactHostRandomMode};
use crate::builtins::common::random::RngAlgorithm;
use crate::context::host_allocation::{HostAllocationPlan, HostAllocationReceipt};
use crate::context::{RuntimeContext, RuntimeManagedWholeWorkSelection, RuntimeWholeWorkSelection};
use crate::{build_runtime_error, BuiltinResult, GpuGatherRetry};
use runmat_accelerate_api::observation::ProviderObservationCapability;
use runmat_execution::{
    single_candidate_resources_admitted, CandidateCostEvidence, CandidateCostUnavailable,
    CandidateExecutionLocation, CandidateOutputPresentation, CandidateOutputResidency,
    CandidatePreparationState, CandidateResourceDemand, CandidateResourceFacts, Digest,
    ExecutionCandidateKind, PlacementResourceSnapshot, PlacementSignatureRef,
    ProviderResourceSnapshot, ResourceByteBound, WholeWorkCandidate, WholeWorkConstraint,
    WholeWorkPlacementRequestRef,
};
use runmat_value::Value;

pub(super) const OPERATION: &str = "stochastic_evolution";

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub(super) enum StochasticRoute {
    Host,
    Provider,
}

pub(super) struct StochasticPlacement {
    pub(super) route: StochasticRoute,
    pub(super) selected: Option<StochasticPlacementSelection>,
}

pub(super) enum StochasticPlacementSelection {
    Portable(RuntimeWholeWorkSelection),
    Managed(RuntimeManagedWholeWorkSelection),
}

pub(super) struct ProviderCandidate<'a> {
    pub(super) plan: &'a StochasticProviderPlan,
    pub(super) resources: &'a StochasticProviderResources,
    pub(super) noise_host_bytes: u64,
}

fn invalid_decision(message: &'static str) -> crate::RuntimeError {
    build_runtime_error(message)
        .with_identifier("RunMat:Placement:InsufficientResourceEvidence")
        .with_gpu_gather_retry(GpuGatherRetry::Never)
        .build()
}

fn resource_legal(candidate: &WholeWorkCandidate, snapshot: &PlacementResourceSnapshot) -> bool {
    single_candidate_resources_admitted(
        CandidateResourceFacts {
            location: candidate.execution_location,
            output_residency: candidate.output_residency,
            presentation: &candidate.presentation,
            kind: candidate.kind,
            demand: candidate.resources,
            transactional_results: candidate.transactional_results,
        },
        snapshot,
        false,
    )
    .is_ok()
}

/// Call only after preparing every offered route, including its initial
/// transfer, logical noise, VM publication and physical completion bounds.
/// Missing evidence removes a speculative optimization before source effects;
/// it never authorizes a narrower executor after a selected route fails.
pub(super) fn select_source(
    context: &RuntimeContext,
    prepared: &StochasticPreparedSource,
    host: Option<&StochasticHostResources>,
    readback: Option<&StochasticHostReadback>,
    provider: Option<ProviderCandidate<'_>>,
    algorithm: RngAlgorithm,
) -> BuiltinResult<Option<StochasticPlacement>> {
    StochasticPreparedSource::checkpoint_receipt(prepared.receipt(), context)?;
    // A context with neither resource nor placement ports is explicitly
    // unmanaged. It has no CPU grant to fabricate and no observation owner;
    // its already-prepared host body can execute without a policy decision.
    // A configured resource port with zero capacity is a different case.
    if context.service_ports().parallel().is_none() && context.service_ports().placement().is_none()
    {
        return Ok(host
            .filter(|_| !matches!(prepared.state(), Value::GpuTensor(_)) || readback.is_some())
            .map(|_| StochasticPlacement {
                route: StochasticRoute::Host,
                selected: None,
            }));
    }
    let parallel = context
        .service_ports()
        .parallel()
        .map(|port| port.placement_resources())
        .unwrap_or_default();
    // Each unique device resource fact is a POD slot. Admit its requested Vec
    // backing before materializing it; the receipt outlives the Vec and
    // selection. No program/operation String is cloned on the managed path.
    let provider_device = provider
        .as_ref()
        .map(|route| route.plan.provider.device_id());
    let readback_device_is_distinct =
        readback.is_some_and(|route| Some(route.provider.device_id()) != provider_device);
    let provider_count = usize::from(provider.is_some()) + usize::from(readback_device_is_distinct);
    let snapshot_receipt = if provider_count != 0 {
        let bytes = std::mem::size_of::<ProviderResourceSnapshot>()
            .checked_mul(provider_count)
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| invalid_decision("provider snapshot size overflow"))?;
        let plan = HostAllocationPlan::new(bytes, 0)
            .map_err(|_| invalid_decision("provider snapshot size overflow"))?;
        match HostAllocationReceipt::admit(Some(context), plan) {
            Ok(receipt) => receipt,
            Err(source) if source.is_capacity_denial() => return Ok(None),
            Err(source) => {
                return Err(
                    build_runtime_error("stochastic provider snapshot was not admitted")
                        .with_identifier(format!(
                            "{}:{}",
                            context.error_namespace(),
                            source.runtime_identifier_suffix()
                        ))
                        .with_source(source)
                        .with_gpu_gather_retry(GpuGatherRetry::Never)
                        .build(),
                );
            }
        }
    } else {
        None
    };
    let mut provider_snapshot = Vec::new();
    if provider_count != 0 {
        provider_snapshot
            .try_reserve_exact(provider_count)
            .map_err(|_| invalid_decision("provider snapshot allocation failed"))?;
    }
    if let Some(route) = provider.as_ref() {
        provider_snapshot.push(route.plan.provider.placement_resources());
    }
    if let Some(route) = readback {
        if !provider_snapshot
            .iter()
            .any(|snapshot| snapshot.device_id == route.provider.device_id())
        {
            provider_snapshot.push(route.provider.placement_resources());
        }
    }
    let snapshot = PlacementResourceSnapshot {
        cpu_millicores_available: parallel.cpu_millicores_available,
        memory_available_bytes: parallel.memory_available_bytes,
        cancellation_requested: context.is_cancelled(),
        providers: provider_snapshot,
        epoch: parallel.epoch,
    };
    let facts = profile(context, prepared, algorithm);
    let provider_required = matches!(prepared.state(), Value::GpuTensor(handle)
        if runmat_accelerate_api::handle_provenance(handle)
            != Some(runmat_accelerate_api::GpuHandleProvenance::Automatic));
    let host_scratch = match readback {
        // Readback query/wrapper/callback metadata is already reserved by
        // its preparation receipt, and this snapshot reports that ledger's
        // remaining capacity. Only the selected native transfer has not yet
        // been granted; quote it here and admit it before any source effect.
        Some(route) => crate::context::host_presentation::combine_host_scratch(
            host.map_or(ResourceByteBound::Known(0), |resources| {
                resources.scratch_bytes
            }),
            route.host_workspace,
        )?,
        None => host.map_or(ResourceByteBound::Known(0), |resources| {
            resources.scratch_bytes
        }),
    };
    let host = host
        .filter(|_| !provider_required)
        .map(|host| WholeWorkCandidate {
            kind: ExecutionCandidateKind::SharedRuntime,
            execution_location: CandidateExecutionLocation::Host,
            preparation: CandidatePreparationState::Ready,
            output_residency: CandidateOutputResidency::Host,
            presentation: CandidateOutputPresentation::Direct,
            route_facts: Digest::sha256_serialized(&(
                StochasticRoute::Host,
                facts,
                host.result_bytes,
                host.scratch_bytes,
                readback.map(|route| (&route.query, route.provider.device_id())),
            ))
            .expect("typed stochastic host route serializes"),
            cost: CandidateCostEvidence::Unavailable(CandidateCostUnavailable::NoEstimate),
            resources: CandidateResourceDemand {
                cpu_millicores: 1_000,
                retained_bytes: host.result_bytes,
                scratch_bytes: host_scratch,
                provider_host_workspace: None,
                provider_input_stages: None,
                queue_slots: 0,
                peak_live: None,
            },
            // t/Z/S are source-ordered checkpoints, not one rollback transaction.
            transactional_results: false,
        })
        .map(|mut candidate| {
            if let Some(route) = readback {
                candidate.presentation = CandidateOutputPresentation::ProviderDownload {
                    device_id: route.provider.device_id(),
                    provider_scratch_bytes: route.provider_scratch,
                    queue_slots: route.queue_slots,
                };
            }
            candidate
        })
        .filter(|candidate| resource_legal(candidate, &snapshot));
    let provider_candidate = provider.as_ref().and_then(|route| {
        if !matches!(
            route
                .plan
                .provider
                .operation_observation_capability(&route.plan.query),
            ProviderObservationCapability::Supported(_)
        ) {
            return None;
        }
        let physical = route.plan.selected.proposal;
        let candidate = WholeWorkCandidate {
            kind: ExecutionCandidateKind::ProviderFusion,
            execution_location: CandidateExecutionLocation::Provider {
                device_id: route.plan.provider.device_id(),
            },
            preparation: CandidatePreparationState::Ready,
            output_residency: CandidateOutputResidency::ProviderAndHost {
                device_id: route.plan.provider.device_id(),
                // Z remains a complete host value, independently of the
                // bounded uploaded tiles. Its quote comes from its owner.
                host_bytes: ResourceByteBound::Known(route.noise_host_bytes),
            },
            presentation: CandidateOutputPresentation::Direct,
            route_facts: Digest::sha256_serialized(&(
                StochasticRoute::Provider,
                facts,
                route.plan.selected.plan.noise_tile_elements,
                physical.frozen_workgroup_size,
                physical.frozen_tile_capacity,
                route.resources.demand,
            ))
            .expect("typed stochastic provider route serializes"),
            cost: CandidateCostEvidence::Unavailable(CandidateCostUnavailable::NoEstimate),
            resources: route.resources.demand,
            transactional_results: false,
        };
        resource_legal(&candidate, &snapshot).then_some(candidate)
    });
    // At most two routes: keep the candidate array on the stack instead of
    // cloning query descriptors or allocating a candidate Vec before selection.
    let (candidates, count) = match (host, provider_candidate) {
        (Some(host), Some(provider)) => ([host, provider], 2),
        (Some(host), None) => ([host.clone(), host], 1),
        (None, Some(provider)) => ([provider.clone(), provider], 1),
        (None, None) => return Ok(None),
    };
    let candidates = &candidates[..count];
    let Some(service) = context.service_ports().placement() else {
        return Ok(candidates
            .iter()
            .find(|candidate| !candidate.kind.is_provider())
            .map(|_| StochasticPlacement {
                route: StochasticRoute::Host,
                selected: None,
            }));
    };
    let revision = match provider.as_ref() {
        Some(route) => service.whole_work_revision_ref(
            context.program_revision(),
            Digest::sha256_serialized(&(
                route.plan.provider.capability_snapshot(),
                readback.map(|source| source.provider.capability_snapshot()),
            ))
            .expect("provider capabilities serialize"),
        ),
        None => match readback {
            Some(route) => service.whole_work_revision_ref(
                context.program_revision(),
                Digest::sha256_serialized(&route.provider.capability_snapshot())
                    .expect("provider capabilities serialize"),
            ),
            None => service.whole_work_host_service_revision_ref(context.program_revision(), facts),
        },
    };
    let Some(revision) = revision else {
        return Ok(None);
    };
    let signature = PlacementSignatureRef {
        region: None,
        operation: OPERATION,
        runtime_facts: facts,
        provider_provenance: None,
        graph_output_authority: None,
        revision,
    };
    let constraint = match prepared.state() {
        Value::GpuTensor(handle)
            if runmat_accelerate_api::handle_provenance(handle)
                != Some(runmat_accelerate_api::GpuHandleProvenance::Automatic) =>
        {
            WholeWorkConstraint::ProviderRequired {
                device_id: handle.device_id(),
            }
        }
        _ => WholeWorkConstraint::AnyLegal,
    };
    let request = WholeWorkPlacementRequestRef {
        signature,
        constraint,
        candidates,
        resources: &snapshot,
        deterministic: true,
        require_transactional_results: false,
    };
    let managed = host_random_mode(Some(context)) == ExactHostRandomMode::Managed
        || snapshot_receipt.is_some();
    let (selected, candidate) = if managed {
        let selected = service.select_whole_work_borrowed_managed(request)?;
        let identity = signature
            .permit_identity()
            .map_err(|_| invalid_decision("stochastic source identity is invalid"))?;
        if selected.identity != identity || selected.selected.node.as_str() != OPERATION {
            return Err(invalid_decision(
                "stochastic selection lost its admitted source identity",
            ));
        }
        let candidate = candidates
            .iter()
            .find(|candidate| {
                selected.selected.candidate == candidate.identity_digest()
                    && selected.selected.kind == candidate.kind
            })
            .ok_or_else(|| invalid_decision("stochastic selection chose an unadmitted route"))?;
        (StochasticPlacementSelection::Managed(selected), candidate)
    } else {
        let selected = service.select_whole_work_borrowed(request)?;
        if selected.decision.signature != signature.to_owned() {
            return Err(invalid_decision(
                "stochastic selection lost its admitted source profile",
            ));
        }
        let candidate = candidates
            .iter()
            .find(|candidate| selected.decision.selected == candidate.selected(OPERATION))
            .ok_or_else(|| invalid_decision("stochastic selection chose an unadmitted route"))?;
        (StochasticPlacementSelection::Portable(selected), candidate)
    };
    StochasticPreparedSource::checkpoint_receipt(prepared.receipt(), context)?;
    StochasticPreparedSource::checkpoint_receipt(snapshot_receipt.as_ref(), context)?;
    Ok(Some(StochasticPlacement {
        route: if candidate.kind.is_provider() {
            StochasticRoute::Provider
        } else {
            StochasticRoute::Host
        },
        selected: Some(selected),
    }))
}
