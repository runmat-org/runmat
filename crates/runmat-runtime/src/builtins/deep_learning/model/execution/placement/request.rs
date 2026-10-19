//! Pure candidate construction from an already admitted host network plan.
//! Shared placement alone chooses among the executable routes.

use std::collections::BTreeSet;

use runmat_accelerate_api::{
    handle_is_explicit, provider, provider_for_device, registered_providers,
    same_provider_instance, GpuHandleProvenance,
};
use runmat_execution::{
    CandidateCostEvidence, CandidateCostUnavailable, CandidateExecutionLocation,
    CandidateOutputResidency, CandidatePreparationState, CandidateResourceDemand, Digest,
    ExecutionCandidateKind, PlacementResourceSnapshot, PlacementSignature,
    ResourceBoundUnavailable, ResourceByteBound, WholeWorkCandidate, WholeWorkConstraint,
    WholeWorkPlacementRequest,
};
use runmat_value::Value;

use super::PendingWholeWorkObservation;
use crate::builtins::deep_learning::model::{
    admission::forward_data,
    gpu::{self, AdmittedHostRoute},
    plan::{
        axes::AxisPlan,
        host::HostNetworkPlan,
        network_workload::{NetworkInvocationFacts, NetworkWorkload, PayloadOrigin},
    },
};
use crate::builtins::deep_learning::{
    deep_learning_error, model::execution::request::NetworkExecutionRequest,
};
use crate::BuiltinResult;

pub(in crate::builtins::deep_learning) enum SelectedNetworkRoute<'h, 'n> {
    Host(PendingWholeWorkObservation),
    Provider {
        route: AdmittedHostRoute<'h, 'n>,
        observation: Option<PendingWholeWorkObservation>,
        present_resident: bool,
    },
}

fn unknown_demand(cpu_millicores: u32, queue_slots: u32) -> CandidateResourceDemand {
    CandidateResourceDemand {
        cpu_millicores,
        retained_bytes: ResourceByteBound::Unavailable(
            ResourceBoundUnavailable::NoRetainedValueBound,
        ),
        scratch_bytes: ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoScratchBound),
        provider_host_workspace: None,
        provider_input_stages: None,
        queue_slots,
        peak_live: None,
    }
}

fn route_digest(consumed: Digest, route: Digest, preparation: Option<Digest>) -> Digest {
    let mut bytes = b"runmat.network.forward.route.v1\0".to_vec();
    bytes.extend_from_slice(consumed.bytes());
    bytes.extend_from_slice(route.bytes());
    match preparation {
        Some(preparation) => {
            bytes.push(1);
            bytes.extend_from_slice(preparation.bytes());
        }
        None => bytes.push(0),
    }
    Digest::sha256(bytes)
}

/// A runtime service compares legal host/provider routes. With no service,
/// ordinary host work remains host; a sole provider route required by an
/// explicit resident learnable executes directly without invented feedback.
pub(in crate::builtins::deep_learning) fn select<'h, 'n>(
    request: &NetworkExecutionRequest<'n>,
    plan: &'h HostNetworkPlan<'n>,
    target_axes: &'h AxisPlan,
    active_tape: bool,
) -> BuiltinResult<Option<SelectedNetworkRoute<'h, 'n>>> {
    let service = request
        .runtime
        .and_then(|runtime| runtime.service_ports().placement());
    let common = NetworkWorkload::from_admitted(plan, target_axes)?;
    let invocation = NetworkInvocationFacts {
        caller: request.caller,
        output_layout: request.output_layout,
        requested_outputs: request.requested_outputs,
        record_tape: active_tape,
    };
    let payloads = std::iter::once(&common.input)
        .chain(common.parameters.iter().map(|row| &row.payload))
        .chain(common.state.iter().map(|row| &row.payload));
    let mut required_device = None;
    for payload in payloads {
        if let PayloadOrigin::Provider {
            allocation,
            provenance: Some(GpuHandleProvenance::Explicit),
            ..
        } = payload.origin
        {
            let device = allocation.device_id();
            if required_device
                .replace(device)
                .is_some_and(|prior| prior != device)
            {
                return Err(deep_learning_error(
                    plan.function(),
                    "explicit resident network values belong to different devices",
                ));
            }
        }
    }
    let host_legal = required_device.is_none();
    if service.is_none() && host_legal {
        return Ok(None);
    }
    let mut candidates = Vec::new();
    let mut routes = Vec::new();
    let mut resident_presentations = Vec::new();
    if host_legal {
        candidates.push(WholeWorkCandidate {
            kind: ExecutionCandidateKind::SharedRuntime,
            execution_location: CandidateExecutionLocation::Host,
            preparation: CandidatePreparationState::Ready,
            output_residency: CandidateOutputResidency::Host,
            presentation: runmat_execution::CandidateOutputPresentation::Direct,
            route_facts: Digest::sha256(b"runmat.network.forward.host.v1"),
            cost: CandidateCostEvidence::Unavailable(CandidateCostUnavailable::NoEstimate),
            resources: unknown_demand(1_000, 0),
            transactional_results: true,
        });
    }
    let mut provider_snapshots = Vec::new();
    if !active_tape {
        let mut available = Vec::new();
        if let Some(active) = provider() {
            available.push(active);
        }
        available.extend(registered_providers());
        let mut visited_devices = BTreeSet::new();
        for provider in available {
            if required_device.is_some_and(|device| device != provider.device_id()) {
                continue;
            }
            if !provider_for_device(provider.device_id())
                .is_some_and(|authorized| same_provider_instance(provider, authorized))
                || !visited_devices.insert(provider.device_id())
            {
                continue;
            }
            let admitted = match gpu::admit_host_route(plan, target_axes, provider) {
                Ok(Some(Ok(route))) => route,
                Ok(None | Some(Err(_))) => continue,
                Err(error) => {
                    if !host_legal {
                        return Err(error);
                    }
                    tracing::warn!(target: "runmat_placement", "provider network admission failed: {error}");
                    continue;
                }
            };
            let selected_rows = admitted.consumed_parameter_views();
            let present_resident = selected_rows.iter().any(|row| {
                matches!(row.value.map(forward_data), Some(Value::GpuTensor(handle)) if handle_is_explicit(handle))
            });
            let selected_work =
                NetworkWorkload::from_selected(plan, target_axes, &selected_rows, &[])?;
            let physical = admitted.physical_route()?.ok_or_else(|| {
                deep_learning_error(
                    plan.function(),
                    "admitted provider route lost its physical stages",
                )
            })?;
            let preparation = provider.deep_learning_forward_preparation_snapshot(&physical);
            let route_facts = route_digest(
                selected_work.semantic_timing_digest(invocation),
                admitted.route_facts,
                preparation.as_ref().map(|p| p.identity),
            );
            let capability = provider.capability_snapshot();
            provider_snapshots.push(capability.clone());
            candidates.push(WholeWorkCandidate {
                kind: ExecutionCandidateKind::ProviderGraph,
                execution_location: CandidateExecutionLocation::Provider {
                    device_id: provider.device_id(),
                },
                preparation: preparation
                    .and_then(|snapshot| snapshot.state)
                    .unwrap_or(CandidatePreparationState::Unknown),
                output_residency: if present_resident {
                    CandidateOutputResidency::Provider {
                        device_id: provider.device_id(),
                    }
                } else {
                    CandidateOutputResidency::Host
                },
                presentation: runmat_execution::CandidateOutputPresentation::Direct,
                route_facts,
                cost: CandidateCostEvidence::Unavailable(CandidateCostUnavailable::NoEstimate),
                resources: unknown_demand(0, 1),
                transactional_results: capability.concurrency.transactional_results,
            });
            routes.push(admitted);
            resident_presentations.push(present_resident);
        }
    }
    if candidates.is_empty() {
        return Err(deep_learning_error(
            plan.function(),
            "no host or provider route preserves the network's resident intent",
        ));
    }
    let Some(runtime) = request.runtime else {
        return required_provider_without_service(plan, routes, resident_presentations);
    };
    let Some(service) = service else {
        return required_provider_without_service(plan, routes, resident_presentations);
    };
    let provider_digest = Digest::sha256(
        serde_json::to_vec(&provider_snapshots)
            .expect("provider capability snapshots must serialize"),
    );
    let Some(revision) =
        service.whole_work_revision(runtime.program_revision().cloned(), provider_digest)
    else {
        return if host_legal {
            Ok(None)
        } else {
            required_provider_without_service(plan, routes, resident_presentations)
        };
    };
    let parallel = runtime
        .service_ports()
        .parallel()
        .map(|port| port.placement_resources())
        .unwrap_or_default();
    let providers = routes
        .iter()
        .map(|route| route.provider().placement_resources())
        .collect::<Vec<_>>();
    let epoch_facts = serde_json::to_vec(&(parallel.epoch, &providers))
        .expect("resource snapshots must serialize");
    let epoch_digest = Digest::sha256(epoch_facts);
    let resource_epoch = u64::from_le_bytes(
        epoch_digest.bytes()[..8]
            .try_into()
            .expect("digest has eight epoch bytes"),
    );
    let resources = PlacementResourceSnapshot {
        cpu_millicores_available: parallel.cpu_millicores_available,
        memory_available_bytes: parallel.memory_available_bytes,
        cancellation_requested: runtime.is_cancelled(),
        providers,
        epoch: resource_epoch,
    };
    let signature = PlacementSignature {
        region: None,
        operation: request.caller.function().into(),
        runtime_facts: common.semantic_timing_digest(invocation),
        provider_provenance: None,
        graph_output_authority: None,
        revision,
    };
    let chosen = service.select_whole_work(WholeWorkPlacementRequest {
        signature: signature.clone(),
        constraint: WholeWorkConstraint::AnyLegal,
        candidates: candidates.clone(),
        resources,
        deterministic: true,
        require_transactional_results: false,
    })?;
    if chosen.decision.signature != signature {
        return Err(deep_learning_error(
            plan.function(),
            "placement selected a different network workload",
        ));
    }
    let selected = candidates
        .iter()
        .position(|candidate| candidate.selected(&signature.operation) == chosen.decision.selected)
        .ok_or_else(|| {
            deep_learning_error(
                plan.function(),
                "placement selected a route outside the admitted network",
            )
        })?;
    let observation = PendingWholeWorkObservation::start(runtime, chosen);
    if host_legal && selected == 0 {
        return Ok(Some(SelectedNetworkRoute::Host(observation)));
    }
    let provider_index = selected - usize::from(host_legal);
    Ok(Some(SelectedNetworkRoute::Provider {
        route: routes.swap_remove(provider_index),
        observation: Some(observation),
        present_resident: resident_presentations.swap_remove(provider_index),
    }))
}

fn required_provider_without_service<'h, 'n>(
    plan: &HostNetworkPlan<'n>,
    mut routes: Vec<AdmittedHostRoute<'h, 'n>>,
    mut resident_presentations: Vec<bool>,
) -> BuiltinResult<Option<SelectedNetworkRoute<'h, 'n>>> {
    if routes.len() != 1 {
        return Err(deep_learning_error(
            plan.function(),
            if routes.is_empty() {
                "no provider route can execute the explicit resident network"
            } else {
                "explicit resident network has multiple provider routes without placement authority"
            },
        ));
    }
    Ok(Some(SelectedNetworkRoute::Provider {
        route: routes.pop().expect("one admitted provider route"),
        observation: None,
        present_resident: resident_presentations
            .pop()
            .expect("one presentation intent"),
    }))
}
